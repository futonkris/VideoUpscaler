import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model.network import build_model
from model.warp import SimpleFlowEstimator
from model.raft import RaftFlow
from data.dataset import build_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SR quality")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--onnx_model", type=str, default=None)
    parser.add_argument("--flow_estimator", type=str, default="raft", choices=["simple", "raft"])
    parser.add_argument("--dataset", type=str, default="vimeo90k", choices=["vimeo90k", "reds"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--mid_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit eval samples")
    parser.add_argument("--save_images", action="store_true", help="Save SR output images")
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

def compute_metrics(sr: np.ndarray, hr: np.ndarray) -> dict[str, float]:
    psnr = peak_signal_noise_ratio(hr, sr, data_range=1.0)
    ssim = structural_similarity(hr, sr, data_range=1.0, channel_axis=2)

    psnr_r = peak_signal_noise_ratio(hr[:, :, 0], sr[:, :, 0], data_range=1.0)
    psnr_g = peak_signal_noise_ratio(hr[:, :, 1], sr[:, :, 1], data_range=1.0)
    psnr_b = peak_signal_noise_ratio(hr[:, :, 2], sr[:, :, 2], data_range=1.0)

    return {
        "psnr": psnr,
        "ssim": ssim,
        "psnr_r": psnr_r,
        "psnr_g": psnr_g,
        "psnr_b": psnr_b,
    }

def compute_bicubic_baseline(lr: np.ndarray, hr: np.ndarray, scale: int) -> dict[str, float]:
    from PIL import Image

    lr_pil = Image.fromarray((lr * 255).astype(np.uint8))
    h, w = hr.shape[:2]
    bicubic = np.array(lr_pil.resize((w, h), Image.BICUBIC)).astype(np.float32) / 255.0
    return compute_metrics(bicubic, hr)

@torch.no_grad()
def benchmark_pytorch(args):
    device = torch.device(args.device)

    model = build_model(
        mid_channels=args.mid_channels,
        num_res_blocks=args.num_res_blocks,
        scale=args.scale,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    if args.flow_estimator == "raft":
        flow_estimator = RaftFlow().to(device)
    else:
        flow_estimator = SimpleFlowEstimator(num_levels=4).to(device)
    flow_estimator.eval()

    dataloader = build_dataloader(
        args.dataset, args.data_root, args.scale, "test",
        batch_size=1, patch_size=128, num_workers=8,
    )

    return run_evaluation(model, flow_estimator, dataloader, device, args)

def benchmark_onnx(args):
    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx_model, providers=["CPUExecutionProvider"])

    dataloader = build_dataloader(
        args.dataset, args.data_root, args.scale, "test",
        batch_size=1, patch_size=128, num_workers=8,
    )

    if args.flow_estimator == "raft":
        flow_estimator = RaftFlow()
    else:
        flow_estimator = SimpleFlowEstimator(num_levels=4)
    flow_estimator.eval()

    all_metrics = {"sr": [], "bicubic": []}
    inference_times = []

    max_samples = args.max_samples or len(dataloader)
    pbar = tqdm(dataloader, total=min(max_samples, len(dataloader)), desc="Benchmarking (ONNX)")

    for i, batch in enumerate(pbar):
        if i >= max_samples:
            break

        lr_curr = batch["lr_curr"]
        lr_prev = batch["lr_prev"]
        lr_next = batch["lr_next"]
        hr_curr = batch["hr_curr"]
        hr_prev = batch["hr_prev"]

        flow_lr_prev = flow_estimator(lr_curr, lr_prev)
        flow_lr_next = flow_estimator(lr_curr, lr_next)

        ort_inputs = {
            "current_lr": lr_curr.numpy(),
            "prev_sr": hr_prev.numpy(),
            "next_lr": lr_next.numpy(),
            "flow_lr_prev": flow_lr_prev.numpy(),
            "flow_lr_next": flow_lr_next.numpy(),
        }

        t0 = time.perf_counter()
        sr_output = sess.run(None, ort_inputs)[0]
        inference_times.append(time.perf_counter() - t0)

        sr_img = np.clip(sr_output[0].transpose(1, 2, 0), 0, 1)
        hr_img = hr_curr[0].numpy().transpose(1, 2, 0)
        lr_img = lr_curr[0].numpy().transpose(1, 2, 0)

        all_metrics["sr"].append(compute_metrics(sr_img, hr_img))
        all_metrics["bicubic"].append(compute_bicubic_baseline(lr_img, hr_img, args.scale))

    return all_metrics, inference_times

def run_evaluation(model, flow_estimator, dataloader, device, args):
    all_metrics = {"sr": [], "bicubic": []}
    inference_times = []

    max_samples = args.max_samples or len(dataloader)
    pbar = tqdm(dataloader, total=min(max_samples, len(dataloader)), desc="Benchmarking")

    for i, batch in enumerate(pbar):
        if i >= max_samples:
            break

        lr_curr = batch["lr_curr"].to(device)
        lr_prev = batch["lr_prev"].to(device)
        lr_next = batch["lr_next"].to(device)
        hr_curr = batch["hr_curr"].to(device)
        hr_prev = batch["hr_prev"].to(device)

        flow_lr_prev = flow_estimator(lr_curr, lr_prev)
        flow_lr_next = flow_estimator(lr_curr, lr_next)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        sr_output = model(lr_curr, hr_prev, lr_next, flow_lr_prev, flow_lr_next)

        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - t0)

        sr_img = sr_output[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        hr_img = hr_curr[0].cpu().numpy().transpose(1, 2, 0)
        lr_img = lr_curr[0].cpu().numpy().transpose(1, 2, 0)

        all_metrics["sr"].append(compute_metrics(sr_img, hr_img))
        all_metrics["bicubic"].append(compute_bicubic_baseline(lr_img, hr_img, args.scale))

        if args.save_images:
            save_comparison(sr_img, hr_img, lr_img, i, args.output_dir)

    return all_metrics, inference_times

def save_comparison(sr: np.ndarray, hr: np.ndarray, lr: np.ndarray, idx: int, output_dir: str):
    from PIL import Image

    out_dir = Path(output_dir) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray((sr * 255).astype(np.uint8)).save(out_dir / f"{idx:04d}_sr.png")
    Image.fromarray((hr * 255).astype(np.uint8)).save(out_dir / f"{idx:04d}_hr.png")

    h, w = hr.shape[:2]
    lr_up = np.array(
        Image.fromarray((lr * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    )
    Image.fromarray(lr_up).save(out_dir / f"{idx:04d}_lr_nearest.png")

def print_results(all_metrics: dict, inference_times: list[float]):
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for method in ["sr", "bicubic"]:
        metrics = all_metrics[method]
        if not metrics:
            continue

        avg_psnr = np.mean([m["psnr"] for m in metrics])
        avg_ssim = np.mean([m["ssim"] for m in metrics])
        std_psnr = np.std([m["psnr"] for m in metrics])
        std_ssim = np.std([m["ssim"] for m in metrics])

        label = "Model SR" if method == "sr" else "Bicubic"
        print(f"\n{label}:")
        print(f"  PSNR:  {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"  SSIM:  {avg_ssim:.4f} ± {std_ssim:.4f}")

    if all_metrics["sr"] and all_metrics["bicubic"]:
        sr_psnr = np.mean([m["psnr"] for m in all_metrics["sr"]])
        bic_psnr = np.mean([m["psnr"] for m in all_metrics["bicubic"]])
        print(f"\n  PSNR gain over bicubic: +{sr_psnr - bic_psnr:.2f} dB")

    if inference_times:
        times_ms = [t * 1000 for t in inference_times]
        print(f"\nInference Timing ({len(times_ms)} frames):")
        print(f"  Average: {np.mean(times_ms):.1f} ms")
        print(f"  Median:  {np.median(times_ms):.1f} ms")
        print(f"  P95:     {np.percentile(times_ms, 95):.1f} ms")
        print(f"  P99:     {np.percentile(times_ms, 99):.1f} ms")
        print(f"  Min:     {np.min(times_ms):.1f} ms")
        print(f"  Max:     {np.max(times_ms):.1f} ms")
        print(f"  FPS:     {1000 / np.mean(times_ms):.1f}")

    print("=" * 70)

def main():
    args = parse_args()

    if args.checkpoint is None and args.onnx_model is None:
        print("Error: Provide either --checkpoint (PyTorch) or --onnx_model (ONNX)")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.onnx_model:
        all_metrics, inference_times = benchmark_onnx(args)
    else:
        all_metrics, inference_times = benchmark_pytorch(args)

    print_results(all_metrics, inference_times)

    results_path = Path(args.output_dir) / f"results_{args.flow_estimator}_{'onnx' if args.onnx_model else 'pytorch'}.npz"
    np.savez(
        results_path,
        sr_psnr=[m["psnr"] for m in all_metrics["sr"]],
        sr_ssim=[m["ssim"] for m in all_metrics["sr"]],
        bicubic_psnr=[m["psnr"] for m in all_metrics["bicubic"]],
        bicubic_ssim=[m["ssim"] for m in all_metrics["bicubic"]],
        inference_times_ms=[t * 1000 for t in inference_times],
    )
    print(f"saved to {results_path}")


if __name__ == "__main__":
    main()