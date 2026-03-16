"""Gradient-weighted Class Activation Mapping (GradCAM) for model interpretability.

GradCAM produces visual explanations for CNN decisions by computing the gradient
of the target class score with respect to the feature maps of a convolutional layer.
The resulting heatmap highlights image regions that most influenced the prediction.

For cell classification, this helps verify the model attends to biologically
relevant features:
    - Nucleus shape and segmentation (lobulated for neutrophils, round for lymphocytes)
    - Cytoplasmic granules (eosinophilic vs basophilic staining)
    - Cell size and shape (biconcave disk for RBCs, irregular for monocytes)
    - Nuclear-to-cytoplasmic ratio

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """GradCAM implementation for convolutional neural networks.

    Hooks into a target convolutional layer to capture forward activations
    and backward gradients during inference. The heatmap is computed as
    a ReLU-filtered weighted combination of activation maps, where the
    weights are the global-average-pooled gradients.

    Args:
        model: The CNN model to explain.
        target_layer: The convolutional layer to use for activation maps.
            Typically the last conv layer before the classifier head.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        # Register hooks to capture activations and gradients
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        """Forward hook: store the layer's output activations."""
        self._activations = output.detach()

    def _save_gradient(
        self, module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        """Backward hook: store the gradients flowing through the layer."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate a GradCAM heatmap for the given input.

        Args:
            input_tensor: Input image tensor, shape (1, C, H, W).
            target_class: Class index to explain. If None, uses the
                predicted (highest-scoring) class.

        Returns:
            Heatmap as a numpy array of shape (H, W) with values in [0, 1],
            resized to match the input spatial dimensions.
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero all gradients and compute gradient of target class score
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        if self._gradients is None or self._activations is None:
            logger.error("GradCAM hooks failed to capture gradients or activations")
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        # Global average pooling of gradients -> channel importance weights
        # Shape: (1, C, H', W') -> (C,)
        weights = torch.mean(self._gradients, dim=(2, 3)).squeeze(0)

        # Weighted combination of activation maps
        # activations: (1, C, H', W'), weights: (C,)
        activations = self._activations.squeeze(0)  # (C, H', W')
        cam = torch.zeros(
            activations.shape[1:], dtype=activations.dtype, device=activations.device
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU: only keep positive contributions (regions that increase the score)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Resize heatmap to input image dimensions
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove registered hooks from the model (cleanup)."""
        self._forward_hook.remove()
        self._backward_hook.remove()
        logger.debug("GradCAM hooks removed")

    def __del__(self) -> None:
        """Ensure hooks are removed when the object is garbage collected."""
        try:
            self.remove_hooks()
        except Exception:
            pass


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a GradCAM heatmap on the original image.

    Creates a blended visualization where warm colors (red/yellow) indicate
    regions the model considers most important for the classification.

    Args:
        image: Original RGB image, shape (H, W, 3), values in [0, 255] uint8.
        heatmap: GradCAM heatmap, shape (H, W), values in [0, 1].
        alpha: Blending weight for the heatmap (0 = original only, 1 = heatmap only).
        colormap: Matplotlib colormap name for the heatmap.

    Returns:
        Blended RGB image, shape (H, W, 3), uint8.
    """
    import matplotlib.cm as cm

    # Apply colormap to heatmap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Resize heatmap to match image dimensions if needed
    if heatmap_colored.shape[:2] != image.shape[:2]:
        from PIL import Image as PILImage

        heatmap_pil = PILImage.fromarray(heatmap_colored)
        heatmap_pil = heatmap_pil.resize((image.shape[1], image.shape[0]), PILImage.LANCZOS)
        heatmap_colored = np.array(heatmap_pil)

    # Blend
    overlay = (1 - alpha) * image.astype(np.float32) + alpha * heatmap_colored.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay
