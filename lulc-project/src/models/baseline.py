"""
Single-backbone baseline for the comparison the paper actually needs.

Why this exists: `rgb_branch_only` is a fair *internal* control, but reviewers
know Helber et al. (2019)'s ResNet-50 at 98.57% on EuroSAT, and that number was
obtained on their splits, not ours. Quoting it next to our accuracy is not a
like-for-like comparison. Training the same backbone family on OUR folds is.

The wrapper deliberately mimics DualBranchEfficientNet's output dict so the
existing training loop, calibration, metrics and compare_variants.py all work
unchanged -- one code path, one evaluation protocol, no second implementation to
keep in sync.

`logits_spectral` is returned as an alias of `logits_rgb` purely to satisfy that
interface. Runners MUST set `branch_loss_weight_spectral=0.0`, otherwise
compute_loss counts the same loss twice and silently doubles the effective
learning signal relative to the dual-branch runs.
"""
import timm
import torch
import torch.nn as nn


class SingleBackboneBaseline(nn.Module):
    """One backbone, RGB only, no fusion -- the conventional approach this project is measured against."""

    def __init__(self, num_classes: int, backbone_name: str = "resnet50",
                 pretrained: bool = True, in_chans: int = 3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes,
        )
        self.backbone_name = backbone_name
        # Same buffer name/shape as the dual-branch model so checkpoints and the
        # calibration call site behave identically.
        self.register_buffer("temperature_rgb", torch.tensor(1.0))
        self.register_buffer("temperature_spectral", torch.tensor(1.0))

    def set_temperatures(self, t_rgb: float, t_spectral: float = None) -> None:
        self.temperature_rgb.fill_(t_rgb)
        self.temperature_spectral.fill_(t_rgb)   # single branch: one temperature

    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor = None) -> dict:
        """`spectral` is accepted and ignored, so the training loop needs no special-casing."""
        logits = self.backbone(rgb)
        probs = torch.softmax(logits / self.temperature_rgb, dim=-1)
        return {
            "logits_rgb": logits,
            "logits_spectral": logits,      # alias; weight it 0 in the loss
            "fused_probs": probs,           # no fusion happens -- this is just the softmax
            "weight_rgb": torch.ones(logits.shape[0], device=logits.device),
            "weight_spectral": torch.zeros(logits.shape[0], device=logits.device),
        }
