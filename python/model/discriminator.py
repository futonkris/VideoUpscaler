import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# U net disc based off how Esrgan does it - spectral normalized
class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        self.skip_connection = skip_connection

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), 0.2, inplace=True)
        out = self.conv9(out)
        return out

def build_discriminator(num_in_ch: int = 3, num_feat: int = 64) -> UNetDiscriminatorSN:
    return UNetDiscriminatorSN(num_in_ch=num_in_ch, num_feat=num_feat)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = build_discriminator().to(device)
    num_params = sum(p.numel() for p in d.parameters())
    print(f"discriminator params {num_params:,}")

    x = torch.randn(2, 3, 384, 384, device=device)
    with torch.no_grad():
        out = d(x)
    print(f"output shape {out.shape}")
    assert out.shape == (2, 1, 384, 384), "shape wrong"
    print("forward pass ok")