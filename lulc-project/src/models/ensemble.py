"""
Dual-Branch EfficientNet-B0 with ECA attention and confidence-weighted
fusion. This is the proposed model from docs/BLUEPRINT.md.

Architecture
------------
Input: a Sentinel-2 patch split into
  - RGB branch input:            (B, 3,   H, W)
  - Spectral-index branch input:  (B, 3+I, H, W)   RGB channels + I index maps (e.g. NDVI)

Each branch:
  timm EfficientNet-B0 backbone (ImageNet-pretrained stem/blocks, first
  conv adapted to extra input channels for the spectral branch)
    -> final feature map (B, 1280, H', W')
    -> ECA channel attention (src/models/attention.py)
    -> global average pool -> (B, 1280)
    -> linear classifier head -> per-branch logits (B, K)

Fusion:
  per-branch softmax -> confidence-weighted fusion (src/models/fusion_torch.py)
  -> fused probability vector (B, K), used for the final prediction.

Both branches' own logits are also returned so the training loop can
supervise each branch directly (each branch must be independently
competent; the ensemble should not rely on one branch carrying the other),
which is also what makes the per-branch ablations in docs/BLUEPRINT.md
possible (e.g. "RGB branch alone" is exactly this same code path with the
spectral branch switched off).
"""
import timm
import torch
import torch.nn as nn

from src.models.attention import ECALayer
from src.models.fusion_torch import confidence_weighted_fusion


ATTENTION_MODES = ("eca", "none")


class _Branch(nn.Module):
    def __init__(self, in_chans: int, num_classes: int, backbone_name: str = "efficientnet_b0",
                 pretrained: bool = True, attention: str = "eca"):
        super().__init__()
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {attention!r}")
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, in_chans=in_chans, num_classes=0, global_pool="",
        )
        feat_channels = self.backbone.num_features
        # nn.Identity for the C3 ablation: keeps the forward path and parameter
        # layout otherwise identical, so the only difference measured is ECA itself.
        self.attention = ECALayer(feat_channels) if attention == "eca" else nn.Identity()
        self.attention_mode = attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feat_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)   # (B, C, H', W')
        feats = self.attention(feats)
        pooled = self.pool(feats).flatten(1)          # (B, C)
        return self.classifier(pooled)                 # (B, K)


class DualBranchEfficientNet(nn.Module):
    def __init__(self, num_classes: int, num_spectral_indices: int = 1, pretrained: bool = True,
                 spectral_in_chans: int | None = None, attention: str = "eca"):
        """
        Args:
            num_classes: number of LULC classes.
            num_spectral_indices: number of extra index channels appended to
                RGB for the spectral branch (e.g. 1 for NDVI alone, 2 for
                NDVI+NDWI). Set to 0 to make the spectral branch RGB-only,
                which is useful for the "no spectral fusion" ablation.
            pretrained: use ImageNet-pretrained backbone weights.
            spectral_in_chans: input channels for the spectral branch. Defaults to
                3 + num_spectral_indices (the blueprint's RGB+index branch). Pass an
                explicit value for branch-input ablations where the spectral branch
                does NOT receive the RGB channels -- see
                src/data/dataset.py::spectral_in_chans.
            attention: "eca" (default, contribution C3) or "none" for the C3 ablation.
        """
        super().__init__()
        self.branch_rgb = _Branch(in_chans=3, num_classes=num_classes, pretrained=pretrained,
                                  attention=attention)
        spec_chans = 3 + num_spectral_indices if spectral_in_chans is None else spectral_in_chans
        if spec_chans < 1:
            raise ValueError(f"spectral branch needs >=1 input channel, got {spec_chans}")
        self.spectral_in_chans = spec_chans
        self.branch_spectral = _Branch(in_chans=spec_chans, num_classes=num_classes,
                                       pretrained=pretrained, attention=attention)
        self.attention_mode = attention
        # Per-branch calibration temperatures (C4). Default 1.0 = no-op, so
        # the model behaves identically to an uncalibrated one until
        # `set_temperatures()` is called after fitting T on a validation
        # split post-training. Stored as buffers (not parameters) so they
        # move with .to(device) and are saved/loaded with the checkpoint,
        # but are never touched by the optimizer or backward().
        self.register_buffer("temperature_rgb", torch.tensor(1.0))
        self.register_buffer("temperature_spectral", torch.tensor(1.0))

    def set_temperatures(self, t_rgb: float, t_spectral: float) -> None:
        """Set post-hoc calibration temperatures fit on a validation split (see src/calibration/temperature_scaling.py)."""
        self.temperature_rgb.fill_(t_rgb)
        self.temperature_spectral.fill_(t_spectral)

    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor) -> dict:
        """
        Args:
            rgb: (B, 3, H, W)
            spectral: (B, C_spec, H, W) where C_spec = self.spectral_in_chans
        Returns:
            dict with:
              logits_rgb, logits_spectral: (B, K) per-branch RAW logits (for per-branch
                supervision -- loss functions should always use these, never the
                calibrated probabilities, so temperature has no effect on gradients)
              fused_probs: (B, K) confidence-weighted fused probabilities, computed from
                *calibrated* per-branch probabilities (temperature_rgb/temperature_spectral)
                -- this is the C1+C4 coupling described in docs/BLUEPRINT.md Section 4:
                fusion weights are only as trustworthy as the confidence they're built from.
              weight_rgb, weight_spectral: (B,) per-sample fusion weights (for analysis/plots)
        """
        logits_rgb = self.branch_rgb(rgb)
        logits_spectral = self.branch_spectral(spectral)

        probs_rgb = torch.softmax(logits_rgb / self.temperature_rgb, dim=-1)
        probs_spectral = torch.softmax(logits_spectral / self.temperature_spectral, dim=-1)
        fused_probs, w_rgb, w_spectral = confidence_weighted_fusion(probs_rgb, probs_spectral)

        return {
            "logits_rgb": logits_rgb,
            "logits_spectral": logits_spectral,
            "fused_probs": fused_probs,
            "weight_rgb": w_rgb,
            "weight_spectral": w_spectral,
        }
