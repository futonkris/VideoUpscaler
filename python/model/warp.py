import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

def warp_with_flow(
    img: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border"
) -> torch.Tensor:
    B, _, H, W = img.shape

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=img.dtype, device=img.device),
        torch.arange(W, dtype=img.dtype, device=img.device),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1) + flow[:, 0]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1) + flow[:, 1]

    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0

    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)

def resize_flow(flow: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, H, W = flow.shape
    resized = F.interpolate(flow, size=(target_h, target_w), mode="bilinear", align_corners=False)
    resized[:, 0] *= target_w / W  
    resized[:, 1] *= target_h / H  
    return resized

class SpyNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 7, padding=3),
        )

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ref, supp, ref - supp, ref + supp], dim=1)[:, :8])

class SimpleFlowEstimator(nn.Module):
    def __init__(self, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.blocks = nn.ModuleList([SpyNetBlock() for _ in range(num_levels)])

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        B, _, H, W = ref.shape

        ref_pyr = [ref]
        supp_pyr = [supp]
        for _ in range(self.num_levels - 1):
            ref_pyr.append(F.avg_pool2d(ref_pyr[-1], 2))
            supp_pyr.append(F.avg_pool2d(supp_pyr[-1], 2))

        flow = torch.zeros(B, 2, H // (2 ** (self.num_levels - 1)),
                           W // (2 ** (self.num_levels - 1)),
                           device=ref.device, dtype=ref.dtype)

        for level in range(self.num_levels - 1, -1, -1):
            ref_l = ref_pyr[level]
            supp_l = supp_pyr[level]

            _, _, h_l, w_l = ref_l.shape

            if flow.shape[2] != h_l or flow.shape[3] != w_l:
                flow = resize_flow(flow, h_l, w_l)

            supp_warped = warp_with_flow(supp_l, flow)

            residual = self.blocks[level](ref_l, supp_warped)
            flow = flow + residual

        return flow

def read_flo(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        assert magic == 202021.25, f"Invalid .flo file: {path}"
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=h * w * 2)
        return flow.reshape((h, w, 2))

def write_flo(flow: np.ndarray, path: str | Path):
    h, w, _ = flow.shape
    with open(path, "wb") as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([w, h], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)

def flow_to_color(flow: np.ndarray, max_flow: float | None = None) -> np.ndarray:
    import colorsys

    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)

    if max_flow is None:
        max_flow = max(mag.max(), 1e-5)

    mag_norm = np.clip(mag / max_flow, 0, 1)

    hue = (angle + np.pi) / (2 * np.pi)  
    hsv = np.stack([hue, np.ones_like(hue), mag_norm], axis=-1)

    rgb = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    for y in range(flow.shape[0]):
        for x in range(flow.shape[1]):
            r, g, b = colorsys.hsv_to_rgb(hsv[y, x, 0], hsv[y, x, 1], hsv[y, x, 2])
            rgb[y, x] = [int(r * 255), int(g * 255), int(b * 255)]

    return rgb