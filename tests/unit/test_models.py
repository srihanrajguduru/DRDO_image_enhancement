import pytest
import torch
import torch.nn as nn
from src.models.lowlight_retinexformer import TwoStageLowLightWrapper
from src.models.dehaze import DehazeFormerWrapper
from src.models.derain import RestormerWrapper
from src.core.config import get_config

class DummyModel(nn.Module):
    def forward(self, x):
        return x

class DummyRetinexFormer(nn.Module):
    def forward(self, x):
        # RetinexFormer returns a tuple (enhanced, illumination) usually
        return x, x


class CaptureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return x

@pytest.fixture(autouse=True)
def setup_cpu():
    get_config().pipeline.device = "cpu"

def test_lowlight_wrapper(monkeypatch):
    wrapper = TwoStageLowLightWrapper()
    
    # Mock load to avoid downloading weights
    def mock_load():
        wrapper.retinexformer = DummyRetinexFormer()
        wrapper.restormer = DummyModel()
        wrapper.is_loaded = True
        
    monkeypatch.setattr(wrapper, "load", mock_load)
    
    img = torch.rand(1, 3, 256, 256)
    out = wrapper.restore(img)
    assert out.shape == img.shape

def test_dehaze_wrapper(monkeypatch):
    wrapper = DehazeFormerWrapper()
    
    def mock_load():
        wrapper.model = DummyModel()
        wrapper.is_loaded = True
        
    monkeypatch.setattr(wrapper, "load", mock_load)
    
    img = torch.rand(1, 3, 256, 256)
    out = wrapper.restore(img)
    assert out.shape == img.shape


def test_dehaze_wrapper_normalization(monkeypatch):
    wrapper = DehazeFormerWrapper()
    capture_model = CaptureModel()

    def mock_load():
        wrapper.model = capture_model
        wrapper.is_loaded = True

    monkeypatch.setattr(wrapper, "load", mock_load)

    img = torch.tensor(
        [[
            [[0.0, 1.0], [0.5, 0.25]],
            [[0.0, 1.0], [0.5, 0.25]],
            [[0.0, 1.0], [0.5, 0.25]],
        ]],
        dtype=torch.float32,
    )

    out = wrapper.restore(img)
    assert capture_model.last_input is not None
    assert torch.isclose(capture_model.last_input.min(), torch.tensor(-1.0))
    assert torch.isclose(capture_model.last_input.max(), torch.tensor(1.0))
    assert torch.allclose(out, img)

def test_derain_wrapper(monkeypatch):
    wrapper = RestormerWrapper()
    
    def mock_load():
        wrapper.model = DummyModel()
        wrapper.is_loaded = True
        
    monkeypatch.setattr(wrapper, "load", mock_load)
    
    img = torch.rand(1, 3, 256, 256)
    out = wrapper.restore(img)
    assert out.shape == img.shape
