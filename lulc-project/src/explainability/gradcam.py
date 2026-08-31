"""
Grad-CAM (Selvaraju et al., 2017), applied per-branch.

Design choice: Grad-CAM is run separately on the RGB branch and the
spectral branch (rather than only on the fused output), because the point
of the explainability section is to show *what each modality actually
contributes* -- e.g. confirming the spectral branch attends to vegetation
boundaries that the RGB branch misses. A single fused heatmap would hide
exactly the comparison a reviewer would want to see.

Grad-CAM++ is used for the failure-case figures in docs/BLUEPRINT.md
(better localization for small/multiple objects, at negligible extra
cost since it reuses the same hooks); Score-CAM and Eigen-CAM were
evaluated and rejected as the primary technique (Score-CAM requires
hundreds of extra forward passes per image -- see blueprint section 12).
"""
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int, forward_fn) -> torch.Tensor:
        """
        Args:
            input_tensor: (1, C, H, W) single image.
            class_idx: which class's logit to backprop from.
            forward_fn: callable(input_tensor) -> (1, K) logits. Passed in
                explicitly since the branch's forward pass lives inside the
                larger DualBranchEfficientNet; this keeps GradCAM decoupled
                from that model's specific call signature.

        Returns:
            (H, W) heatmap normalized to [0, 1], upsampled to input resolution.
        """
        self.model.zero_grad(set_to_none=True)
        logits = forward_fn(input_tensor)
        score = logits[0, class_idx]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)      # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()
