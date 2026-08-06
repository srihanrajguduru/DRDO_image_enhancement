"""
Two-Stage Low-Light Enhancement Pipeline:
  Stage 1: RetinexFormer — restores brightness and structure
  Stage 2: Restormer (real denoising) — cleans residual noise

This achieves significantly higher PSNR than either model alone.
"""

import torch
from src.models.base import BaseModel
from src.core.config import get_models_config, get_config
from src.utils.optimization import optimize_model, infer_with_autocast
from src.utils.logger import get_logger

logger = get_logger(__name__)

from src.models.retinexformer_arch import RetinexFormer
from src.models.restormer_arch import Restormer


class TwoStageLowLightWrapper(BaseModel):
    """
    Two-stage low-light enhancement:
      1. RetinexFormer for brightness restoration
      2. Restormer (real denoising) for residual noise cleanup
    """

    def __init__(self):
        super().__init__("TwoStageLowLight")
        self.retinex_config = get_models_config().retinexformer
        self.denoise_config = get_models_config().restormer_denoise
        self.sys_config = get_config().pipeline
        self.stage1_model = None
        self.stage2_model = None

    def _load_retinexformer(self):
        """Load Stage 1: RetinexFormer."""
        params = self.retinex_config.params
        model = RetinexFormer(
            in_channels=params.get("in_channels", 3),
            out_channels=params.get("out_channels", 3),
            n_feat=params.get("n_feat", 31),
            stage=params.get("stage", 1),
            num_blocks=params.get("num_blocks", [1, 2, 2]),
        )

        checkpoint = torch.load(
            self.retinex_config.checkpoint_path,
            map_location="cpu", weights_only=False,
        )
        if "params" in checkpoint:
            state_dict = checkpoint["params"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip "module." prefix if present
        clean_sd = {
            k.replace("module.", "", 1): v for k, v in state_dict.items()
        }
        model.load_state_dict(clean_sd)
        logger.info(f"Loaded RetinexFormer from {self.retinex_config.checkpoint_path}")
        return model

    def _load_restormer_denoise(self):
        """Load Stage 2: Restormer real denoising."""
        model = Restormer(
            inp_channels=3, out_channels=3, dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="BiasFree",
        )

        checkpoint = torch.load(
            self.denoise_config.checkpoint_path,
            map_location="cpu", weights_only=False,
        )
        if "params" in checkpoint:
            state_dict = checkpoint["params"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        clean_sd = {
            k.replace("module.", "", 1): v for k, v in state_dict.items()
        }
        model.load_state_dict(clean_sd)
        logger.info(
            f"Loaded Restormer (denoising) from {self.denoise_config.checkpoint_path}"
        )
        return model

    def load(self) -> None:
        if self.is_loaded:
            return

        logger.info("Loading two-stage low-light pipeline...")

        try:
            self.stage1_model = self._load_retinexformer()
            self.stage1_model = optimize_model(
                self.stage1_model,
                device=self.sys_config.device,
                precision=self.sys_config.precision,
                compile_model=self.sys_config.compile_models,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load RetinexFormer weights: {e}. "
                "Using random init — download LOL_v2_real.pth to weights/"
            )
            self.stage1_model = None

        try:
            self.stage2_model = self._load_restormer_denoise()
            self.stage2_model = optimize_model(
                self.stage2_model,
                device=self.sys_config.device,
                precision=self.sys_config.precision,
                compile_model=self.sys_config.compile_models,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load Restormer denoising weights: {e}. "
                "Stage 2 (denoising) will be skipped."
            )
            self.stage2_model = None

        self.is_loaded = True

    def warmup(self) -> None:
        if not self.is_loaded:
            self.load()
        dummy = torch.randn(1, 3, 256, 256, device=self.sys_config.device)
        _ = self.restore(dummy)

    def restore(self, image: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            self.load()

        x = image.to(self.sys_config.device)

        # Stage 1: RetinexFormer — brightness restoration
        if self.stage1_model is not None:
            x = infer_with_autocast(
                self.stage1_model, x,
                precision=self.sys_config.precision,
                device=self.sys_config.device,
            )
            x = torch.clamp(x, 0.0, 1.0).float()

        # Stage 2: Restormer — residual noise removal (optional via config)
        if self.sys_config.lowlight_denoise and self.stage2_model is not None:
            x = infer_with_autocast(
                self.stage2_model, x,
                precision=self.sys_config.precision,
                device=self.sys_config.device,
            )
            x = torch.clamp(x, 0.0, 1.0).float()

        return x
