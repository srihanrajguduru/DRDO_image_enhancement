from typing import Dict, List
import numpy as np
from src.detectors.base import BaseDetector
from src.core.config import get_config


class DegradationRouter:
    """Routes an image to the appropriate restoration models based on detected degradations."""

    def __init__(self, detectors: Dict[str, BaseDetector]):
        self.detectors = detectors
        self.config = get_config().router
        self.thresholds = {
            "haze": self.config.haze_threshold,
            "lowlight": self.config.lowlight_threshold,
            "rain": self.config.rain_threshold,
        }

    def detect_all(self, image: np.ndarray) -> Dict[str, float]:
        """Runs all configured detectors and returns their confidence scores."""
        if "learned" in self.detectors:
            return self.detectors["learned"].detect_all(image)

        scores = {}
        for name, detector in self.detectors.items():
            scores[name] = detector.detect(image)
        return scores

    def route(self, scores: Dict[str, float]) -> List[str]:
        """
        Determines the execution plan (which models to run in which order).

        Applies mutual-exclusion rules to resolve detector conflicts:
        - Haze + Rain: suppress rain unless its score dominates haze by >0.3,
          because haze images with vertical structures (buildings) often trigger
          the rain detector as a false positive.
        """
        active_degradations = []
        for deg in self.config.execution_order:
            if deg in scores and scores[deg] > self.thresholds.get(deg, 0.5):
                active_degradations.append(deg)

        # Mutual exclusion: haze and rain rarely co-occur at similar intensities.
        # If both fire, keep rain ONLY if it clearly dominates.
        if "haze" in active_degradations and "rain" in active_degradations:
            if scores["rain"] - scores["haze"] < 0.3:
                active_degradations.remove("rain")

        return active_degradations

    def execute_routing(self, image: np.ndarray) -> tuple[Dict[str, float], List[str]]:
        """Convenience method to detect and route in one step."""
        scores = self.detect_all(image)
        plan = self.route(scores)
        return scores, plan
