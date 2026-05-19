import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()

class PerceptualLoss(nn.Module):
    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
        use_input_norm: bool = True,
    ):
        super().__init__()

        if layer_weights is None:
            layer_weights = {
                "conv3_4": 1.0,
                "conv4_4": 1.0,
            }

        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        features = vgg.features

        self.layer_name_to_idx = {
            "conv1_2": 3,
            "conv2_2": 8,
            "conv3_4": 17,
            "conv4_4": 26,
            "conv5_4": 35,
        }

        max_idx = max(self.layer_name_to_idx[name] for name in layer_weights.keys())
        self.vgg_layers = nn.Sequential(*list(features.children())[: max_idx + 1])

        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_norm:
            return (x - self.mean) / self.std
        return x

    def _extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = {}
        for idx, layer in enumerate(self.vgg_layers):
            x = layer(x)
            for name, layer_idx in self.layer_name_to_idx.items():
                if idx == layer_idx and name in self.layer_weights:
                    features[name] = x
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)

        pred_features = self._extract_features(pred_norm)
        target_features = self._extract_features(target_norm)

        loss = torch.tensor(0.0, device=pred.device)
        for name, weight in self.layer_weights.items():
            loss = loss + weight * F.l1_loss(pred_features[name], target_features[name])

        return loss

class CombinedLoss(nn.Module):
    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.01,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight

        self.pixel_loss = CharbonnierLoss()
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pix_loss = self.pixel_loss(pred, target)
        total = self.pixel_weight * pix_loss

        loss_dict = {"pixel_loss": pix_loss.item()}

        if self.perceptual_loss is not None:
            perc_loss = self.perceptual_loss(pred, target)
            total = total + self.perceptual_weight * perc_loss
            loss_dict["perceptual_loss"] = perc_loss.item()

        loss_dict["total_loss"] = total.item()
        return total, loss_dict

class GANLoss(nn.Module):
    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        label = self.real_label if target_is_real else self.fake_label
        target = torch.full_like(pred, label)
        return self.loss(pred, target)