import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, mid_channels: int = 64, num_blocks: int = 15):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = nn.Sequential(
            *[ResidualBlock(mid_channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = self.body(feat) + feat
        return feat

class FlowWarp(nn.Module):
    def forward(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=x.dtype, device=x.device),
            torch.arange(0, W, dtype=x.dtype, device=x.device),
            indexing="ij",
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1) + flow[:, 0, :, :]
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1) + flow[:, 1, :, :]
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

def resize_flow(flow: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, H, W = flow.shape
    resized = F.interpolate(flow, size=(target_h, target_w), mode="bilinear", align_corners=False)
    scale_w = target_w / W
    scale_h = target_h / H
    out = torch.empty_like(resized)
    out[:, 0] = resized[:, 0] * scale_w
    out[:, 1] = resized[:, 1] * scale_h
    return out

# this is just fusing 3 context streams at low res and pixel unshuffle brings previous high res to low res wihout losing information

class BidirectionalFusion(nn.Module):
    def __init__(self, mid_channels: int = 64, scale: int = 2, num_blocks: int = 5):
        super().__init__()
        self.scale = scale
        prev_compressed_c = mid_channels * (scale ** 2)
        in_c = mid_channels + prev_compressed_c + mid_channels
        self.fuse_head = nn.Sequential(
            nn.Conv2d(in_c, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse_body = nn.Sequential(
            *[ResidualBlock(mid_channels) for _ in range(num_blocks)]
        )

    def forward(
        self,
        curr_feat: torch.Tensor,
        prev_feat_hr_warped: torch.Tensor,
        next_feat_warped: torch.Tensor,
    ) -> torch.Tensor:
        prev_compressed = F.pixel_unshuffle(prev_feat_hr_warped, self.scale)
        x = torch.cat([curr_feat, prev_compressed, next_feat_warped], dim=1)
        x = self.fuse_head(x)
        x = self.fuse_body(x) + x 
        return x

class Reconstruction(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 3, scale: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        if scale == 2:
            layers.extend([
                nn.Conv2d(in_channels, in_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
            ])
        elif scale == 3:
            layers.extend([
                nn.Conv2d(in_channels, in_channels * 9, 3, padding=1),
                nn.PixelShuffle(3),
                nn.LeakyReLU(0.1, inplace=True),
            ])
        elif scale == 4:
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(in_channels, in_channels * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.1, inplace=True),
                ])
        else:
            raise ValueError(f"Unsupported scale factor: {scale}. Use 2, 3, or 4.")
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.upscale = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upscale(x)

# comapred to previous model the previous super resed frame does not get downsampled anym before feature extraction and are jst extracted at high res.
# also added forward direction for more context

class TemporalSRNet(nn.Module): 
    def __init__(self, mid_channels: int = 64, num_res_blocks: int = 15, scale: int = 2):
        super().__init__()
        self.scale = scale

        self.feat_lr = FeatureExtractor(
            in_channels=3, mid_channels=mid_channels, num_blocks=num_res_blocks,
        )
        hr_blocks = max(num_res_blocks // 3, 3)
        self.feat_hr = FeatureExtractor(
            in_channels=3, mid_channels=mid_channels, num_blocks=hr_blocks,
        )

        self.warp = FlowWarp()

        fusion_blocks = max(num_res_blocks // 3, 3)
        self.fusion = BidirectionalFusion(
            mid_channels=mid_channels, scale=scale, num_blocks=fusion_blocks,
        )

        self.reconstruction = Reconstruction(
            in_channels=mid_channels, out_channels=3, scale=scale,
        )

    def forward(
        self,
        current_lr: torch.Tensor,
        prev_sr: torch.Tensor,
        next_lr: torch.Tensor,
        flow_lr_prev: torch.Tensor,
        flow_lr_next: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = current_lr.shape

        curr_feat = self.feat_lr(current_lr)
        next_feat = self.feat_lr(next_lr)
        next_feat_warped = self.warp(next_feat, flow_lr_next)

        prev_feat_hr = self.feat_hr(prev_sr)
        flow_hr_prev = resize_flow(flow_lr_prev, H * self.scale, W * self.scale)
        prev_feat_hr_warped = self.warp(prev_feat_hr, flow_hr_prev)

        fused = self.fusion(curr_feat, prev_feat_hr_warped, next_feat_warped)
        sr_output = self.reconstruction(fused)
        return sr_output

def build_model(
    mid_channels: int = 64,
    num_res_blocks: int = 15,
    scale: int = 2,
) -> TemporalSRNet:
    return TemporalSRNet(
        mid_channels=mid_channels,
        num_res_blocks=num_res_blocks,
        scale=scale,
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(mid_channels=128, num_res_blocks=15, scale=2).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {num_params:,}")

    B, H, W, S = 2, 64, 64, 2
    current_lr = torch.randn(B, 3, H, W, device=device)
    prev_sr = torch.randn(B, 3, H * S, W * S, device=device)
    next_lr = torch.randn(B, 3, H, W, device=device)
    flow_lr_prev = torch.randn(B, 2, H, W, device=device)
    flow_lr_next = torch.randn(B, 2, H, W, device=device)

    with torch.no_grad():
        out = model(current_lr, prev_sr, next_lr, flow_lr_prev, flow_lr_next)
    print(f"output shape: {out.shape}")
    assert out.shape == (B, 3, H * S, W * S), "shapes wrong"
    print("forward pass ok")