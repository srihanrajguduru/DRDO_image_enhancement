from typing import Dict
from src.detectors.base import BaseDetector
from src.detectors.haze import HazeDetector
from src.detectors.lowlight import LowLightDetector
from src.detectors.rain import RainDetector
from src.detectors.learned import LearnedDetector

from src.models.base import BaseModel
from src.models.dehaze import DehazeFormerWrapper
from src.models.lowlight_retinexformer import TwoStageLowLightWrapper
from src.models.derain import RestormerWrapper
from src.core.config import get_config


class ModelRegistry:
    """Registry to lazily initialize and manage models and detectors."""

    def __init__(self):
        self._detectors: Dict[str, BaseDetector] = {}
        self._models: Dict[str, BaseModel] = {}

    def get_detectors(self) -> Dict[str, BaseDetector]:
        if not self._detectors:
            detector_type = get_config().router.detector_type
            if detector_type == "learned":
                self._detectors = {
                    "learned": LearnedDetector(),
                }
            else:
                self._detectors = {
                    "haze": HazeDetector(),
                    "lowlight": LowLightDetector(),
                    "rain": RainDetector(),
                }
        return self._detectors

    def get_model(self, name: str) -> BaseModel:
        if name not in self._models:
            if name == "haze":
                self._models[name] = DehazeFormerWrapper()
            elif name == "lowlight":
                self._models[name] = TwoStageLowLightWrapper()
            elif name == "rain":
                self._models[name] = RestormerWrapper()
            else:
                raise ValueError(f"Unknown model name: {name}")

        return self._models[name]

    def unload_all(self):
        """Unloads all models to free VRAM."""
        for model in self._models.values():
            model.unload()


registry = ModelRegistry()
