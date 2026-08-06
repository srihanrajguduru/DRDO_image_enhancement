import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from src.detectors.base import BaseDetector
from src.core.config import PROJECT_ROOT, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LearnedDetector(BaseDetector):
    """
    Lightweight CNN (MobileNetV2) multi-label detector for haze, lowlight, and rain.
    Predicts confidence scores in [0.0, 1.0] for all degradations in a single forward pass.
    """

    def __init__(self, checkpoint_path: str = "weights/detector.pth"):
        self.checkpoint_path = str(PROJECT_ROOT / checkpoint_path)
        self.device = get_config().pipeline.device
        self.model = None
        self.is_loaded = False
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load(self):
        if self.is_loaded:
            return

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 3)

        if os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded learned detector weights from {self.checkpoint_path}")
        else:
            logger.warning(
                f"Learned detector weights not found at {self.checkpoint_path}. "
                "Using untrained MobileNetV2 architecture."
            )

        model = model.to(self.device)
        model.eval()
        self.model = model
        self.is_loaded = True

    def detect_all(self, image: np.ndarray) -> dict:
        """
        Runs single-pass CNN inference returning dict of scores:
        {'haze': float, 'lowlight': float, 'rain': float}
        """
        if not self.is_loaded:
            self.load()

        pil_img = Image.fromarray(image)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        return {
            "haze": float(probs[0]),
            "lowlight": float(probs[1]),
            "rain": float(probs[2]),
        }

    def detect(self, image: np.ndarray) -> float:
        # Default single detect returns max degradation probability
        scores = self.detect_all(image)
        return max(scores.values())
