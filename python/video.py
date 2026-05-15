import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.network import build_model
from model.raft import RaftFlow

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--mid_channels", type=int, default=64)
    p.add_argument("--num_res_blocks", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true", help="Run SR model in fp16")
    p.add_argument("--crf", type=int, default=16,
                   help="x264 quality (lower=better, 16-20 typical)")
    p.add_argument("--preset", default="medium", help="x264 encoder preset")
    p.add_argument("--no_audio", action="store_true",
                   help="Skip audio passthrough")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only N frames (0 = all)")
    return p.parse_args()

def probe_video(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of", "json", path,
    ]
    info = json.loads(subprocess.check_output(cmd).decode())["streams"][0]
    fps_str = info.get("avg_frame_rate") or info["r_frame_rate"]
    num, den = (int(x) for x in fps_str.split("/"))
    fps = num / den if den else float(num)
    nb = int(info.get("nb_frames", 0) or 0)
    if nb == 0 and "duration" in info:
        nb = int(float(info["duration"]) * fps)
    return {"width": int(info["width"]),
            "height": int(info["height"]),
            "fps": fps, "nb_frames": nb}

def open_reader(path: str) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vsync", "passthrough", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

def open_writer(path: str, w: int, h: int, fps: float,
                crf: int, preset: str, audio_src: str | None) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "-",
    ]
    if audio_src:
        cmd += ["-i", audio_src,
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:a", "copy", "-shortest"]
    cmd += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10**8)

def frame_to_tensor(arr: np.ndarray, device, h: int, w: int) -> torch.Tensor:
    """HWC uint8 -> 1,3,H,W float32 [0,1]."""
    return (torch.from_numpy(arr).to(device, non_blocking=True)
            .permute(2, 0, 1).unsqueeze(0).float() / 255.0)

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    model = build_model(
        mid_channels=args.mid_channels,
        num_res_blocks=args.num_res_blocks,
        scale=args.scale,
    ).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    if args.fp16:
        model = model.half()

    flow_estimator = RaftFlow().to(device).eval()

    info = probe_video(args.input)
    in_w, in_h = info["width"], info["height"]
    out_w, out_h = in_w * args.scale, in_h * args.scale
    fps = info["fps"]
    n_total = info["nb_frames"]
    if args.limit > 0:
        n_total = min(n_total, args.limit) if n_total else args.limit
    print(f"Input: {in_w}x{in_h}")
    print(f"Output: {out_w}x{out_h}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    reader = open_reader(args.input)
    writer = open_writer(
        args.output, out_w, out_h, fps, args.crf, args.preset,
        audio_src=None if args.no_audio else args.input,
    )

    frame_bytes = in_w * in_h * 3

    def read_one_frame() -> np.ndarray | None:
        raw = reader.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            return None
        return np.frombuffer(raw, np.uint8).reshape(in_h, in_w, 3).copy()

    frame_curr_np = read_one_frame()
    if frame_curr_np is None:
        print("Empty video.")
        return
    frame_next_np = read_one_frame()  

    prev_lr_fp32: torch.Tensor | None = None
    prev_sr: torch.Tensor | None = None 

    pbar = tqdm(total=n_total or None, desc="Upscaling", unit="f")
    t0 = time.perf_counter()
    idx = 0

    try:
        while frame_curr_np is not None:
            if args.limit > 0 and idx >= args.limit:
                break

            frame_next_for_model = frame_next_np if frame_next_np is not None else frame_curr_np

            cur_lr_fp32 = frame_to_tensor(frame_curr_np, device, in_h, in_w)
            next_lr_fp32 = frame_to_tensor(frame_next_for_model, device, in_h, in_w)

            if prev_sr is None:
                bicubic = F.interpolate(
                    cur_lr_fp32, scale_factor=args.scale,
                    mode="bicubic", align_corners=False,
                ).clamp(0, 1)
                prev_sr_in = bicubic.half() if args.fp16 else bicubic
                flow_lr_prev = torch.zeros(1, 2, in_h, in_w, device=device)
            else:
                flow_lr_prev = flow_estimator(cur_lr_fp32, prev_lr_fp32)
                prev_sr_in = prev_sr  

            flow_lr_next = flow_estimator(cur_lr_fp32, next_lr_fp32)

            cur_in = cur_lr_fp32.half() if args.fp16 else cur_lr_fp32
            next_in = next_lr_fp32.half() if args.fp16 else next_lr_fp32
            flo_p_in = flow_lr_prev.half() if args.fp16 else flow_lr_prev
            flo_n_in = flow_lr_next.half() if args.fp16 else flow_lr_next

            sr = model(cur_in, prev_sr_in, next_in, flo_p_in, flo_n_in)
            sr = sr.clamp(0, 1)

            prev_sr = sr
            prev_lr_fp32 = cur_lr_fp32

            out_arr = (sr.float().squeeze(0).permute(1, 2, 0)
                       .mul(255.0).add(0.5)
                       .clamp(0, 255).to(torch.uint8)
                       .cpu().numpy())
            writer.stdin.write(out_arr.tobytes())
            idx += 1
            pbar.update(1)

            frame_curr_np = frame_next_np
            frame_next_np = read_one_frame() if frame_next_np is not None else None
    finally:
        pbar.close()
        try:
            writer.stdin.close()
        except Exception:
            pass
        writer.wait()
        reader.wait()

    elapsed = time.perf_counter() - t0
    print(f"{idx} frames in {elapsed:.1f}s "
          f"({idx / max(elapsed, 1e-6):.2f} fps), "
          f"output - {args.output}")


if __name__ == "__main__":
    main()