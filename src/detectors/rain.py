import cv2
import numpy as np
from src.detectors.base import BaseDetector


class RainDetector(BaseDetector):
    """Detects rain streaks using vertical edge analysis and high-frequency extraction.

    Algorithm:
    1. Extract high-frequency content via Gaussian difference.
    2. Binarize to isolate strong edges.
    3. Morphological opening with a vertical kernel to keep only
       elongated vertical structures (rain streaks).
    4. Compute streak density as fraction of surviving pixels.

    Note: This detector is sensitive by design — it catches light rain but
    may also respond to vertical textures in non-rain images. The routing
    threshold in config.yaml (default: 0.45) acts as the final gate to
    control false positive vs missed rain tradeoff.
    """

    def __init__(self, rain_threshold: float = 0.02):
        self.rain_threshold = rain_threshold

    def _analyze_streaks(self, image: np.ndarray) -> float:
        """Analyzes directional edge gradients to detect rain streaks."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 1. Isolate high-frequency content by subtracting a blurred version.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred)

        # 2. Binarize — keep only pixels with strong high-frequency energy.
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        # 3. Morphological opening with a vertical kernel (1×7).
        #    Keeps only structures at least 7 pixels tall and 1 pixel wide.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        vertical_streaks = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 4. Compute streak density.
        streak_density = np.count_nonzero(vertical_streaks) / vertical_streaks.size

        # Normalize to [0, 1] confidence score.
        score = max(0.0, min(1.0, streak_density / self.rain_threshold))
        return float(score)

    def detect(self, image: np.ndarray) -> float:
        """Returns rain confidence score in [0.0, 1.0]."""
        return self._analyze_streaks(image)
