import cv2
import numpy as np
from src.detectors.base import BaseDetector


class HazeDetector(BaseDetector):
    """Detects haze using dark channel, contrast, and saturation cues."""

    def __init__(self, patch_size: int = 15):
        self.patch_size = patch_size

    def _get_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """Calculates the dark channel of an image."""
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel

    def detect(self, image: np.ndarray) -> float:
        """
        Calculates haze confidence in [0, 1].
        Higher dark-channel intensity, lower contrast, and lower saturation
        increase haze confidence.
        """
        img_float = image.astype(np.float64) / 255.0
        dark_channel = self._get_dark_channel(img_float)

        dark_term = float(np.mean(dark_channel))

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        contrast_term = float(np.clip(1.0 - (np.std(gray) / 0.25), 0.0, 1.0))

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64) / 255.0
        desaturation_term = float(np.clip(1.0 - np.mean(hsv[:, :, 1]), 0.0, 1.0))

        confidence = 0.5 * dark_term + 0.3 * contrast_term + 0.2 * desaturation_term

        return float(np.clip(confidence, 0.0, 1.0))
