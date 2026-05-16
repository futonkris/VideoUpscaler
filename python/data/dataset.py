import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

def _sample_aug() -> dict:
    return {
        "flip_h": random.random() > 0.5,
        "flip_v": random.random() > 0.5,
        "rot90":  random.random() > 0.5,
    }

def _apply_aug_image(img: torch.Tensor, aug: dict) -> torch.Tensor:
    if aug["flip_h"]:
        img = torch.flip(img, [-1])
    if aug["flip_v"]:
        img = torch.flip(img, [-2])
    if aug["rot90"]:
        img = torch.rot90(img, 1, [-2, -1])
    return img

def _apply_aug_flow(flow: torch.Tensor, aug: dict) -> torch.Tensor:
    if aug["flip_h"]:
        flow = torch.flip(flow, [-1]).clone()
        flow[0] = -flow[0]
    if aug["flip_v"]:
        flow = torch.flip(flow, [-2]).clone()
        flow[1] = -flow[1]
    if aug["rot90"]:
        flow = torch.rot90(flow, 1, [-2, -1])
        u, v = flow[0], flow[1]
        flow = torch.stack([-v, u], dim=0)
    return flow

class Vimeo90KDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        scale: int = 4,
        split: str = "train",
        patch_size: int = 64,
        augment: bool = True,
        use_precomputed_flow: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment and split == "train"
        self.split = split
        self.use_precomputed_flow = use_precomputed_flow

        list_file = self.data_root / f"tri_{'train' if split == 'train' else 'test'}list.txt"
        if not list_file.exists():
            raise FileNotFoundError(
                f"file list not found"
                f"download vimeo"
            )
        with open(list_file) as f:
            self.sequences = [line.strip() for line in f if line.strip()]
        print(f"Vimeo-90K {split}: {len(self.sequences)} triplets"
              f"{'(precomputed flow)' if use_precomputed_flow else ''}")

    def __len__(self) -> int:
        return len(self.sequences)

    def _load_frame(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return transforms.ToTensor()(img)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_path = self.data_root / "sequences" / self.sequences[idx]

        hr_frames_full = [
            self._load_frame(seq_path / f"im{i}.png") for i in range(1, 4)
        ]

        lr_frames_full = [
            F.interpolate(
                f.unsqueeze(0), scale_factor=1.0 / self.scale, mode="bicubic",
                align_corners=False, antialias=True,
            ).squeeze(0).clamp(0, 1)
            for f in hr_frames_full
        ]

        flow_back: torch.Tensor | None = None
        flow_fwd: torch.Tensor | None = None
        if self.use_precomputed_flow:
            try:
                flow_back = torch.from_numpy(
                    np.load(seq_path / "back_flow.npy").astype(np.float32)
                )
                flow_fwd = torch.from_numpy(
                    np.load(seq_path / "forward_flow.npy").astype(np.float32)
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"precomputed flow missing for {seq_path}."
                    f"run precompute or set to false"
                )

        _, lr_H, lr_W = lr_frames_full[0].shape
        if self.split == "train":
            top_lr = random.randint(0, lr_H - self.patch_size)
            left_lr = random.randint(0, lr_W - self.patch_size)
        else:
            top_lr = (lr_H - self.patch_size) // 2
            left_lr = (lr_W - self.patch_size) // 2

        top_hr = top_lr * self.scale
        left_hr = left_lr * self.scale
        hr_patch = self.patch_size * self.scale

        lr_frames = [
            f[:, top_lr:top_lr + self.patch_size, left_lr:left_lr + self.patch_size]
            for f in lr_frames_full
        ]
        hr_frames = [
            f[:, top_hr:top_hr + hr_patch, left_hr:left_hr + hr_patch]
            for f in hr_frames_full
        ]
        if flow_back is not None:
            flow_back = flow_back[
                :, top_lr:top_lr + self.patch_size, left_lr:left_lr + self.patch_size
            ]
            flow_fwd = flow_fwd[
                :, top_lr:top_lr + self.patch_size, left_lr:left_lr + self.patch_size
            ]

        if self.augment:
            aug = _sample_aug()
            lr_frames = [_apply_aug_image(f, aug) for f in lr_frames]
            hr_frames = [_apply_aug_image(f, aug) for f in hr_frames]
            if flow_back is not None:
                flow_back = _apply_aug_flow(flow_back, aug)
                flow_fwd = _apply_aug_flow(flow_fwd, aug)

        result = {
            "lr_prev": lr_frames[0],
            "lr_curr": lr_frames[1],
            "lr_next": lr_frames[2],
            "hr_prev": hr_frames[0],
            "hr_curr": hr_frames[1],
            "hr_next": hr_frames[2],
        }
        if flow_back is not None:
            result["flow_lr_prev"] = flow_back
            result["flow_lr_next"] = flow_fwd
        return result

class REDSDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        scale: int = 4,
        split: str = "train",
        patch_size: int = 64,
        num_frames: int = 3,
        augment: bool = True,
        use_precomputed_flow: bool = False,
    ):
        super().__init__()
        if use_precomputed_flow:
            raise NotImplementedError(
                "no precomputed flow for REDS yet run with false"
            )
        self.data_root = Path(data_root)
        self.scale = scale
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.augment = augment and split == "train"
        self.split = split
        self.use_precomputed_flow = False

        sharp_dir = self.data_root / "train_sharp"
        if not sharp_dir.exists():
            raise FileNotFoundError(
                f"REDS not found{sharp_dir}\n"
                f"download reds"
            )

        all_seqs = sorted([d.name for d in sharp_dir.iterdir() if d.is_dir()])
        if split == "train":
            self.sequences = [s for s in all_seqs if int(s) < 240]
        else:
            self.sequences = [s for s in all_seqs if int(s) >= 240]

        self.samples = []
        for seq in self.sequences:
            seq_dir = sharp_dir / seq
            frames = sorted(seq_dir.glob("*.png"))
            num_available = len(frames)
            for start in range(num_available - num_frames + 1):
                self.samples.append((seq, start))

        print(f"REDS {split} {len(self.samples)} samples from {len(self.sequences)} sequences")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq, start = self.samples[idx]
        sharp_dir = self.data_root / "train_sharp" / seq

        hr_frames_full = []
        for i in range(self.num_frames):
            frame_path = sharp_dir / f"{start + i:08d}.png"
            img = Image.open(frame_path).convert("RGB")
            hr_frames_full.append(transforms.ToTensor()(img))

        lr_frames_full = [
            F.interpolate(
                f.unsqueeze(0), scale_factor=1.0 / self.scale, mode="bicubic",
                align_corners=False, antialias=True,
            ).squeeze(0).clamp(0, 1)
            for f in hr_frames_full
        ]

        _, lr_H, lr_W = lr_frames_full[0].shape
        if self.split == "train":
            top_lr = random.randint(0, lr_H - self.patch_size)
            left_lr = random.randint(0, lr_W - self.patch_size)
        else:
            top_lr = (lr_H - self.patch_size) // 2
            left_lr = (lr_W - self.patch_size) // 2

        top_hr = top_lr * self.scale
        left_hr = left_lr * self.scale
        hr_patch = self.patch_size * self.scale

        lr_frames = [
            f[:, top_lr:top_lr + self.patch_size, left_lr:left_lr + self.patch_size]
            for f in lr_frames_full
        ]
        hr_frames = [
            f[:, top_hr:top_hr + hr_patch, left_hr:left_hr + hr_patch]
            for f in hr_frames_full
        ]

        if self.augment:
            aug = _sample_aug()
            lr_frames = [_apply_aug_image(f, aug) for f in lr_frames]
            hr_frames = [_apply_aug_image(f, aug) for f in hr_frames]

        return {
            "lr_prev": lr_frames[0],
            "lr_curr": lr_frames[1],
            "lr_next": lr_frames[2] if self.num_frames > 2 else lr_frames[1],
            "hr_prev": hr_frames[0],
            "hr_curr": hr_frames[1],
            "hr_next": hr_frames[2] if self.num_frames > 2 else hr_frames[1],
        }

def build_dataloader(
    dataset_name: str,
    data_root: str | None,
    scale: int = 4,
    split: str = "train",
    batch_size: int = 8,
    patch_size: int = 64,
    num_workers: int = 8,
    use_precomputed_flow: bool = False,
    vimeo_root: str | None = None,
    reds_root: str | None = None,
) -> torch.utils.data.DataLoader:
    name = dataset_name.lower()
    if name == "combined":
        if vimeo_root is None or reds_root is None:
            raise ValueError(
                "--vimeo_root and --reds_root are required for 'combined' dataset"
            )
        if use_precomputed_flow:
            raise ValueError(
                "combined training cannot use precomputed flow yet"
            )
        ds_vimeo = Vimeo90KDataset(
            data_root=vimeo_root, scale=scale, split=split, patch_size=patch_size,
            use_precomputed_flow=False,
        )
        ds_reds = REDSDataset(
            data_root=reds_root, scale=scale, split=split, patch_size=patch_size,
            use_precomputed_flow=False,
        )
        dataset = torch.utils.data.ConcatDataset([ds_vimeo, ds_reds])
        print(f"combined {split} {len(dataset)} samples ("
              f"{len(ds_vimeo)} vimeo + {len(ds_reds)} REDS)")
    elif name == "vimeo90k":
        dataset = Vimeo90KDataset(
            data_root=data_root, scale=scale, split=split, patch_size=patch_size,
            use_precomputed_flow=use_precomputed_flow,
        )
    elif name == "reds":
        dataset = REDSDataset(
            data_root=data_root, scale=scale, split=split, patch_size=patch_size,
            use_precomputed_flow=use_precomputed_flow,
        )
    else:
        raise ValueError(f"no {dataset_name}. use either vimeo or REDS")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )