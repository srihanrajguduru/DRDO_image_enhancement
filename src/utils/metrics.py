import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import lpips as lpips_lib
import piq
from PIL import Image
import warnings

# Suppress imquality warnings
warnings.filterwarnings("ignore", module="imquality")


class QualityMetrics:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.lpips_vgg = lpips_lib.LPIPS(net="vgg").to(device)
        self.lpips_vgg.eval()

    def calculate_psnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate PSNR. Requires clean reference (target)."""
        return float(psnr_metric(target, pred, data_range=255))

    def calculate_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate SSIM. Requires clean reference (target)."""
        return float(ssim_metric(target, pred, channel_axis=2, data_range=255))

    def calculate_lpips(
        self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> float:
        """Calculate LPIPS (lower is better). Inputs should be [0, 1] tensors."""
        with torch.no_grad():
            # Scale to [-1, 1] for LPIPS
            pred_scaled = pred_tensor * 2.0 - 1.0
            target_scaled = target_tensor * 2.0 - 1.0
            val = self.lpips_vgg(pred_scaled, target_scaled)
        return float(val.item())

    def calculate_brisque(self, pred: np.ndarray) -> float:
        """
        Calculate BRISQUE (No-reference metric, lower is better ~0-100).
        Uses the piq library which is compatible with modern scikit-image/PyTorch.
        """
        try:
            # piq expects a float tensor [0,1] with shape (1, C, H, W)
            pred_t = (
                torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            )
            score = piq.brisque(pred_t, data_range=1.0, reduction="mean")
            return float(score.item())
        except Exception:
            # Fallback: try imquality
            try:
                from imquality import brisque
                pil_img = Image.fromarray(pred)
                score = brisque.score(pil_img)
                return float(score)
            except Exception:
                return -1.0  # -1 means "could not compute"

    def evaluate_all(self, pred: np.ndarray, target: np.ndarray) -> dict:
        """
        Full-reference evaluation. Both pred and target must be clean/ground-truth pair.
        pred  : restored image (H,W,C) uint8
        target: clean ground-truth image (H,W,C) uint8
        """
        pred_t = (
            torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            / 255.0
        )
        target_t = (
            torch.from_numpy(target)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            / 255.0
        )

        return {
            "psnr": self.calculate_psnr(pred, target),
            "ssim": self.calculate_ssim(pred, target),
            "lpips": self.calculate_lpips(pred_t, target_t),
            "brisque": self.calculate_brisque(pred),
        }

    def evaluate_no_reference(self, pred: np.ndarray) -> dict:
        """
        No-reference evaluation — when no ground-truth is available.
        Returns only BRISQUE (the only no-reference metric in this pipeline).
        PSNR/SSIM/LPIPS are set to None since they require a clean reference.
        """
        return {
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "brisque": self.calculate_brisque(pred),
        }