import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def validate_vimeo90k(data_root: Path) -> bool:
    seq_dir = data_root / "sequences"
    train_list = data_root / "tri_trainlist.txt"
    test_list = data_root / "tri_testlist.txt"

    if not seq_dir.exists():
        print(f"missing {seq_dir}")
        print("download vimeo")
        return False

    if not train_list.exists() or not test_list.exists():
        print(f"missing the training/test folders in {data_root}")
        return False

    groups = list(seq_dir.iterdir())
    total_seqs = sum(1 for g in groups for _ in g.iterdir() if g.is_dir())
    print(f"vimeo ok {total_seqs} triplets in {len(groups)} groups")

    with open(train_list) as f:
        sample = f.readline().strip()
    sample_path = seq_dir / sample
    if sample_path.exists():
        frames = list(sample_path.glob("*.png"))
        print(f"  Sample: {sample_path} — {len(frames)} frames")
        if frames:
            img = Image.open(frames[0])
            print(f"  Resolution: {img.size[0]}x{img.size[1]}")
    return True

def validate_reds(data_root: Path) -> bool:
    sharp_dir = data_root / "train_sharp"

    if not sharp_dir.exists():
        print(f"missing {sharp_dir}")
        print("download REDS")
        return False

    sequences = sorted([d for d in sharp_dir.iterdir() if d.is_dir()])
    print(f"REDS ok {len(sequences)} sequences")

    if sequences:
        frames = sorted(sequences[0].glob("*.png"))
        print(f"{sequences[0].name} — {len(frames)} frames")
        if frames:
            img = Image.open(frames[0])
            print(f"resolution {img.size[0]}x{img.size[1]}")
    return True

def compute_dataset_stats(data_root: Path, dataset: str, max_samples: int = 500):
    to_tensor = transforms.ToTensor()
    pixel_sum = torch.zeros(3)
    pixel_sq_sum = torch.zeros(3)
    num_pixels = 0

    if dataset == "vimeo90k":
        with open(data_root / "tri_trainlist.txt") as f:
            sequences = [line.strip() for line in f if line.strip()][:max_samples]
        for seq in tqdm(sequences, desc="computing stats"):
            for i in range(1, 4):
                path = data_root / "sequences" / seq / f"im{i}.png"
                if path.exists():
                    img = to_tensor(Image.open(path).convert("RGB"))
                    pixel_sum += img.sum(dim=[1, 2])
                    pixel_sq_sum += (img ** 2).sum(dim=[1, 2])
                    num_pixels += img.shape[1] * img.shape[2]
    elif dataset == "reds":
        sharp_dir = data_root / "train_sharp"
        sequences = sorted([d for d in sharp_dir.iterdir() if d.is_dir()])[:10]
        for seq_dir in tqdm(sequences, desc="computing stats"):
            frames = sorted(seq_dir.glob("*.png"))[:50]
            for path in frames:
                img = to_tensor(Image.open(path).convert("RGB"))
                pixel_sum += img.sum(dim=[1, 2])
                pixel_sq_sum += (img ** 2).sum(dim=[1, 2])
                num_pixels += img.shape[1] * img.shape[2]

    mean = pixel_sum / num_pixels
    std = torch.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
    print(f"Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    stats_path = data_root / "dataset_stats.npz"
    np.savez(stats_path, mean=mean.numpy(), std=std.numpy())
    print(f"saved to {stats_path}")


def precompute_flow_placeholder(data_root: Path, dataset: str):
    pass

def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--dataset", type=str, required=True, choices=["vimeo90k", "reds"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--compute_stats", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"Data root does not exist: {data_root}")
        print("\nPlease download the dataset first:")
        if args.dataset == "vimeo90k":
            print("wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip")
            print("unzip vimeo_triplet.zip -d ../data/vimeo90k/")
        else:
            print("Visit https://seungjunnah.github.io/Datasets/reds.html")
            print("Download train_sharp and extract to ../data/REDS/")
        sys.exit(1)

    if args.dataset == "vimeo90k":
        valid = validate_vimeo90k(data_root)
    else:
        valid = validate_reds(data_root)

    if not valid:
        sys.exit(1)

    if args.compute_stats:
        compute_dataset_stats(data_root, args.dataset)

    precompute_flow_placeholder(data_root, args.dataset)

    print("data prep complete")

if __name__ == "__main__":
    main()