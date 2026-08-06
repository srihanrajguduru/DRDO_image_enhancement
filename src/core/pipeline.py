import time
from typing import Dict, Any, Union, Optional
from pathlib import Path

from src.core.config import get_config
from src.core.registry import registry
from src.core.router import DegradationRouter
from src.utils.logger import get_logger
from src.utils.image import (
    load_image,
    save_image,
    numpy_to_tensor,
    tensor_to_numpy,
    resize_for_inference,
    unpad_inference,
)
from src.utils.metrics import QualityMetrics
from src.utils.tta import self_ensemble

logger = get_logger(__name__)


class RestorationPipeline:
    """End-to-End Image Restoration Pipeline."""

    def __init__(self):
        self.config = get_config()
        self.router = DegradationRouter(registry.get_detectors())
        self.metrics = QualityMetrics(device=self.config.pipeline.device)

    def process_image(
        self,
        image_path: Union[str, Path],
        save_path: Union[str, Path] = None,
        reference_path: Union[str, Path] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single image.
        1. Loads image (and clean reference, if provided).
        2. Detects degradations.
        3. Routes to models.
        4. Applies models sequentially (with optional self-ensemble TTA).
        5. Computes metrics.
        6. Saves restored image (if save_path is provided).

        Args:
            image_path: Path to the degraded input image.
            save_path: Optional path to save the restored output.
            reference_path: Optional path to a clean ground-truth image,
                paired with image_path. Required for full-reference metrics
                (PSNR/SSIM/LPIPS) to be meaningful. Without it, only
                no-reference metrics are computed.
        """
        start_time = time.perf_counter()

        # 1. Load Image (+ optional clean reference)
        logger.info(f"Processing {image_path}")
        original_img_np = load_image(image_path)

        clean_img_np = None
        if reference_path is not None:
            logger.info(f"Loading ground-truth reference {reference_path}")
            clean_img_np = load_image(reference_path)

        # 2. Detect and Route
        scores, execution_plan = self.router.execute_routing(original_img_np)
        logger.info(f"Degradation scores: {scores}")
        logger.info(f"Execution plan: {execution_plan}")

        # Fast path if no degradations
        if not execution_plan:
            logger.info(
                "No degradations detected above threshold. Skipping restoration."
            )
            if save_path:
                save_image(original_img_np, save_path)
            return {
                "scores": scores,
                "plan": [],
                "metrics": self._compute_metrics(original_img_np, clean_img_np),
                "latency_ms": (time.perf_counter() - start_time) * 1000,
            }

        # 3. Prepare tensor
        img_tensor = numpy_to_tensor(
            original_img_np, device=self.config.pipeline.device
        )
        img_tensor, original_size = resize_for_inference(img_tensor, multiple=16)

        # 4. Apply models (with optional self-ensemble TTA)
        use_tta = self.config.pipeline.self_ensemble
        for deg_type in execution_plan:
            logger.info(f"Applying model for: {deg_type}" + (" (with TTA)" if use_tta else ""))
            model = registry.get_model(deg_type)
            if use_tta:
                img_tensor = self_ensemble(model.restore, img_tensor)
            else:
                img_tensor = model.restore(img_tensor)

        # Unpad and convert back to numpy
        img_tensor = unpad_inference(img_tensor, original_size)
        restored_img_np = tensor_to_numpy(img_tensor)

        # 5. Compute metrics
        quality_metrics = self._compute_metrics(restored_img_np, clean_img_np)

        # 6. Save if needed
        if save_path:
            save_image(restored_img_np, save_path)

        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Finished processing in {latency:.2f} ms")

        return {
            "scores": scores,
            "plan": execution_plan,
            "metrics": quality_metrics,
            "latency_ms": latency,
        }

    def _compute_metrics(
        self, restored_img_np, clean_img_np: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Full-reference metrics (PSNR/SSIM/LPIPS) require a clean ground-truth
        image and are only computed when one is available (clean_img_np is
        not None). Comparing restored output against the degraded input
        instead of a clean reference produces numbers with no valid meaning,
        so we deliberately never do that.

        Falls back to QualityMetrics.evaluate_no_reference(), if present, so
        that unpaired images (no ground truth) still get a no-reference score
        such as BRISQUE instead of a meaningless full-reference comparison.
        """
        if clean_img_np is not None:
            return self.metrics.evaluate_all(restored_img_np, clean_img_np)

        if hasattr(self.metrics, "evaluate_no_reference"):
            return self.metrics.evaluate_no_reference(restored_img_np)

        logger.warning(
            "No reference_path was provided and QualityMetrics has no "
            "evaluate_no_reference() method yet — skipping metrics for this "
            "image rather than comparing restored output against the "
            "degraded input."
        )
        return {}