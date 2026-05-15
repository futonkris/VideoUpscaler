import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models.optical_flow import raft_large
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="vimeo90k", choices=["vimeo90k", "reds"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--batch_size", type=int, default=8,
                   help="Triplets per RAFT batch. Increase if you've got VRAM.")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute flows even if files already exist")
    return p.parse_args()

_to_tensor = transforms.ToTensor()

def load_frame(path: Path, device) -> torch.Tensor:
    return _to_tensor(Image.open(path).convert("RGB")).to(device)

def gather_triplets(data_root: Path, dataset: str) -> list[Path]:
    if dataset == "vimeo90k":
        return sorted((data_root / "sequences").glob("*/*"))
    elif dataset == "reds":
        raise NotImplementedError("REDS precompute not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def filter_todo(triplets: list[Path], overwrite: bool) -> list[Path]:
    if overwrite:
        return triplets
    todo = []
    for d in triplets:
        if not (d / "back_flow.npy").exists() or not (d / "forward_flow.npy").exists():
            todo.append(d)
    return todo

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    triplets_all = gather_triplets(data_root, args.dataset)
    todo = filter_todo(triplets_all, args.overwrite)
    print(f"{len(triplets_all)} triplets total; {len(todo)} need processing")
    if not todo:
        print("zoo wee mama")
        return

    model = raft_large(weights=models.optical_flow.Raft_Large_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model = model.eval().to(device)
    print("rafting rafted")

    BATCH = args.batch_size
    SCALE = args.scale
    t0 = time.perf_counter()
    done = 0

    pbar = tqdm(range(0, len(todo), BATCH), desc="flow batches")
    for i in pbar:
        batch_dirs = todo[i:i + BATCH]

        refs, currs, futs = [], [], []
        for d in batch_dirs:
            try:
                hr = [load_frame(d / f"im{j}.png", device) for j in (1, 2, 3)]
            except FileNotFoundError as e:
                print(f"\nSkipping {d.name}: {e}")
                continue
            lr = [
                F.interpolate(
                    f.unsqueeze(0), scale_factor=1.0 / SCALE, mode="bicubic",
                    align_corners=False, antialias=True,
                ).squeeze(0).clamp(0, 1)
                for f in hr
            ]
            refs.append(lr[0])
            currs.append(lr[1])
            futs.append(lr[2])

        if not refs:
            continue

        ref = torch.stack(refs, dim=0)
        curr = torch.stack(currs, dim=0)
        fut = torch.stack(futs, dim=0)

        B, _, H, W = ref.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        ref = F.pad(ref, (0, pad_w, 0, pad_h), mode="replicate")
        curr = F.pad(curr, (0, pad_w, 0, pad_h), mode="replicate")
        fut = F.pad(fut, (0, pad_w, 0, pad_h), mode="replicate")

        ref = ref * 2.0 - 1.0
        curr = curr * 2.0 - 1.0
        fut = fut * 2.0 - 1.0

        back_flow = model(curr, ref)[-1][:, :, :H, :W]
        forward_flow = model(curr, fut)[-1][:, :, :H, :W]

        back_np = back_flow.cpu().numpy().astype(np.float16)
        fwd_np = forward_flow.cpu().numpy().astype(np.float16)

        for j, d in enumerate(batch_dirs):
            np.save(d / "back_flow.npy", back_np[j])
            np.save(d / "forward_flow.npy", fwd_np[j])

        done += len(batch_dirs)
        elapsed = time.perf_counter() - t0
        rate = done / max(elapsed, 1e-6)
        pbar.set_postfix(rate=f"{rate:.1f} triplets/s")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {done} triplets in {elapsed/60:.1f} min "
          f"({done/elapsed:.1f} triplets/s)")
    sample = todo[0] / "back_flow.npy"
    if sample.exists():
        size_kb = sample.stat().st_size / 1024
        print(f"sample size: {size_kb:.1f}kb "
              f"total size - {size_kb * 2 * done / 1024 / 1024:.1f}gb")

if __name__ == "__main__":
    main()