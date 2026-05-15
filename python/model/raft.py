import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from pathlib import Path
from torchvision.models.optical_flow import raft_large

class RaftFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = raft_large(weights=models.optical_flow.Raft_Large_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model = self.model.eval()

    def forward (self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        B, _, H, W = ref.shape

        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8

        ref = F.pad(ref, (0, pad_w, 0, pad_h), mode="replicate")
        supp = F.pad(supp, (0, pad_w, 0, pad_h), mode="replicate")

        ref = ref * 2.0 - 1.0
        supp = supp * 2.0 - 1.0

        flows = self.model(ref, supp)

        return flows[-1][:, :, :H, :W]

