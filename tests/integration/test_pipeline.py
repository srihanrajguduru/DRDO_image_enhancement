import pytest
import numpy as np
from pathlib import Path
from src.core.pipeline import RestorationPipeline

@pytest.fixture
def dummy_image(tmp_path):
    img_path = tmp_path / "dummy.jpg"
    # Create a dummy RGB image using numpy and PIL, save it
    from PIL import Image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(img).save(img_path)
    return img_path

def test_pipeline_initialization():
    from src.core.config import get_config
    get_config().pipeline.device = "cpu"
    pipeline = RestorationPipeline()
    assert pipeline is not None
    assert pipeline.router is not None

def test_process_image(dummy_image, tmp_path):
    from src.core.config import get_config
    get_config().pipeline.device = "cpu"
    pipeline = RestorationPipeline()
    # Mock models config to avoid downloading weights during tests
    # We are using dummy architectures in our wrappers which are self-contained.
    
    out_path = tmp_path / "out.jpg"
    results = pipeline.process_image(dummy_image, save_path=out_path)
    
    assert "scores" in results
    assert "plan" in results
    assert "metrics" in results
    assert out_path.exists()
