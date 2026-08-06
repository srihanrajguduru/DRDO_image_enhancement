import torch
from src.models.base import BaseModel
from src.core.config import get_models_config, get_config
from src.utils.optimization import optimize_model, infer_with_autocast
from src.utils.logger import get_logger

logger = get_logger(__name__)

from src.models.dehazeformer_arch import dehazeformer_b


class DehazeFormerWrapper(BaseModel):
    """Wrapper for DehazeFormer for haze removal."""

    def __init__(self):
        super().__init__("DehazeFormer")
        self.config = get_models_config().dehazeformer
        self.sys_config = get_config().pipeline

    def load(self) -> None:
        if self.is_loaded:
            return

        logger.info(f"Loading {self.name} from {self.config.checkpoint_path}")

        # Instantiate model architecture
        self.model = dehazeformer_b()

        # Load weights
        try:
            checkpoint = torch.load(
                self.config.checkpoint_path, map_location="cpu", weights_only=False
            )
            state_dict = (
                checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            )

            # The official checkpoints often have "module." prefix from DataParallel training.
            # We must strip it so it maps correctly to the single-GPU model.
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    clean_state_dict[k[7:]] = v
                else:
                    clean_state_dict[k] = v

            self.model.load_state_dict(clean_state_dict)
        except Exception as e:
            logger.warning(
                f"Could not load weights for {self.name}, using random initialization. Error: {e}"
            )

        # Optimize model (FP16, compile, etc.)
        self.model = optimize_model(
            self.model,
            device=self.sys_config.device,
            precision=self.sys_config.precision,
            compile_model=self.sys_config.compile_models,
        )

        self.is_loaded = True

    def warmup(self) -> None:
        if not self.is_loaded:
            self.load()
        logger.info(f"Warming up {self.name}")
        dummy_input = torch.randn(1, 3, 256, 256, device=self.sys_config.device)
        _ = self.restore(dummy_input)

    def restore(self, image: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            self.load()

        # DehazeFormer checkpoints are trained with normalized inputs in [-1, 1].
        # Convert from pipeline range [0, 1] -> [-1, 1], then map output back.
        image_input = image.to(self.sys_config.device)
        image_input = image_input * 2.0 - 1.0
        
        # Run inference with autocast
        out = infer_with_autocast(
            self.model,
            image_input,
            precision=self.sys_config.precision,
            device=self.sys_config.device,
        )
        
        # Map model output [-1, 1] back to pipeline range [0, 1].
        out = (out + 1.0) * 0.5
        return torch.clamp(out, 0, 1)
