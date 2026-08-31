"""
Efficient Channel Attention (Wang et al., CVPR 2020).

Why ECA over SE / CBAM (see docs/BLUEPRINT.md section 4 for the full
comparison table):
  - SE block: a channel-attention MLP with a dimensionality-reduction
    bottleneck (fc -> reduce -> fc -> expand). The reduction ratio is an
    extra hyperparameter and adds C*C/r parameters.
  - CBAM: SE-style channel attention *plus* a spatial attention branch
    (7x7 conv over pooled feature maps) -- roughly 2x the added params and
    compute of SE alone, for a task (scene classification, not
    localization) where spatial attention has shown smaller gains.
  - ECA: replaces the two fully-connected layers with a single 1D
    convolution of small kernel size k applied directly over the
    channel-pooled descriptor, avoiding the bottleneck entirely. This adds
    only k parameters per attention module (k is typically 3-5) versus the
    thousands added by an SE block's bottleneck MLP.

For a paper whose stated goal is efficiency, ECA is the one attention
mechanism whose added parameter/FLOP cost is close to zero, so it is the
only one kept as a core component. SE/CBAM remain in the ablation study
purely as a comparison point ("what if we spent the extra params anyway"),
not as part of the proposed model.
"""
import math
import torch
import torch.nn as nn


class ECALayer(nn.Module):
    """Efficient Channel Attention. Drop-in module: y = ECALayer()(x) has the same shape as x."""

    def __init__(self, channels: int, gamma: int = 2, beta: int = 1):
        super().__init__()
        # Adaptive kernel size, as proposed in the original ECA paper.
        t = int(abs((math.log2(channels) + beta) / gamma))
        kernel_size = t if t % 2 else t + 1
        kernel_size = max(kernel_size, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)        # (B, 1, C)
        y = self.conv(y)                           # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)      # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
