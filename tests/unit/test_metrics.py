import torch
import pytest
from src.utils.metrics import QualityMetrics

@pytest.fixture
def metrics_engine():
    from src.core.config import get_config
    get_config().pipeline.device = "cpu"
    return QualityMetrics(device="cpu")

@pytest.fixture
def dummy_images():
    import numpy as np
    # 64x64x3 numpy arrays in [0, 255] range
    img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return img1, img2

def test_psnr(metrics_engine, dummy_images):
    img1, img2 = dummy_images
    psnr_val = metrics_engine.calculate_psnr(img1, img2)
    assert isinstance(psnr_val, float)
    
    # Identical images should have very high PSNR
    assert metrics_engine.calculate_psnr(img1, img1) > 80.0

def test_ssim(metrics_engine, dummy_images):
    img1, img2 = dummy_images
    ssim_val = metrics_engine.calculate_ssim(img1, img2)
    assert isinstance(ssim_val, float)
    assert -1.0 <= ssim_val <= 1.0
    
    # Identical images should have SSIM ~ 1.0
    assert metrics_engine.calculate_ssim(img1, img1) > 0.99

def test_brisque(metrics_engine, dummy_images):
    img1, _ = dummy_images
    brisque_val = metrics_engine.calculate_brisque(img1)
    assert isinstance(brisque_val, float)

def test_evaluate_all(metrics_engine):
    import numpy as np
    img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    results = metrics_engine.evaluate_all(img1, img2)
    
    assert "psnr" in results
    assert "ssim" in results
    assert "lpips" in results
    assert "brisque" in results
    assert isinstance(results["psnr"], float)
    assert isinstance(results["ssim"], float)
    assert isinstance(results["brisque"], float)

def test_evaluate_no_reference(metrics_engine):
    import numpy as np
    img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    results = metrics_engine.evaluate_no_reference(img1)
    
    assert "psnr" in results and results["psnr"] is None
    assert "ssim" in results and results["ssim"] is None
    assert "lpips" in results and results["lpips"] is None
    assert "brisque" in results
    assert isinstance(results["brisque"], float)
