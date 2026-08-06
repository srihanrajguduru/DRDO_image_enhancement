"""
Self-Ensemble Test-Time Augmentation (TTA) for image restoration.

Applies 8 geometric transformations (identity, horizontal flip, vertical flip,
both flips, and 4 rotations), runs inference on each, reverses the transforms,
and averages the results. Typically improves PSNR by 0.1–0.5 dB.
"""

import torch
from typing import Callable


def _apply_transform(img: torch.Tensor, transform_id: int) -> torch.Tensor:
    """Apply one of 8 geometric transforms to a (B, C, H, W) tensor."""
    if transform_id == 0:
        return img  # identity
    elif transform_id == 1:
        return torch.flip(img, dims=[3])  # horizontal flip
    elif transform_id == 2:
        return torch.flip(img, dims=[2])  # vertical flip
    elif transform_id == 3:
        return torch.flip(img, dims=[2, 3])  # both flips
    elif transform_id == 4:
        return torch.rot90(img, k=1, dims=[2, 3])  # 90° CW
    elif transform_id == 5:
        return torch.flip(torch.rot90(img, k=1, dims=[2, 3]), dims=[3])  # 90° + H-flip
    elif transform_id == 6:
        return torch.rot90(img, k=2, dims=[2, 3])  # 180°
    elif transform_id == 7:
        return torch.rot90(img, k=3, dims=[2, 3])  # 270° CW
    else:
        raise ValueError(f"Unknown transform_id: {transform_id}")


def _apply_inverse_transform(img: torch.Tensor, transform_id: int) -> torch.Tensor:
    """Reverse one of 8 geometric transforms on a (B, C, H, W) tensor."""
    if transform_id == 0:
        return img
    elif transform_id == 1:
        return torch.flip(img, dims=[3])
    elif transform_id == 2:
        return torch.flip(img, dims=[2])
    elif transform_id == 3:
        return torch.flip(img, dims=[2, 3])
    elif transform_id == 4:
        return torch.rot90(img, k=3, dims=[2, 3])  # inverse of 90° CW
    elif transform_id == 5:
        return torch.flip(torch.rot90(img, k=3, dims=[2, 3]), dims=[3])
    elif transform_id == 6:
        return torch.rot90(img, k=2, dims=[2, 3])
    elif transform_id == 7:
        return torch.rot90(img, k=1, dims=[2, 3])  # inverse of 270° CW
    else:
        raise ValueError(f"Unknown transform_id: {transform_id}")


def self_ensemble(
    restore_fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    num_transforms: int = 8,
) -> torch.Tensor:
    """
    Apply self-ensemble TTA to an image restoration function.

    Args:
        restore_fn: A callable that takes a (B, C, H, W) tensor and returns
                     the restored (B, C, H, W) tensor.
        image: Input tensor (B, C, H, W) in [0, 1].
        num_transforms: Number of transforms to use (4 = flips only, 8 = flips + rotations).
                         Using 4 is faster, 8 is more accurate.

    Returns:
        Averaged restored tensor (B, C, H, W).
    """
    num_transforms = min(num_transforms, 8)
    outputs = []

    for t_id in range(num_transforms):
        # Apply transform
        transformed = _apply_transform(image, t_id)
        # Run inference
        restored = restore_fn(transformed)
        # Reverse transform
        reversed_out = _apply_inverse_transform(restored, t_id)
        outputs.append(reversed_out)

    # Average all outputs
    result = torch.stack(outputs, dim=0).mean(dim=0)
    return result
