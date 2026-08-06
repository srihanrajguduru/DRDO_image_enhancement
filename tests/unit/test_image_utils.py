import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from src.utils.image import load_image, save_image, numpy_to_tensor, tensor_to_numpy, resize_for_inference, unpad_inference

@pytest.fixture
def dummy_image_file(tmp_path):
    img_path = tmp_path / "dummy.png"
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(img).save(img_path)
    return img_path

def test_load_image(dummy_image_file):
    img = load_image(dummy_image_file)
    assert isinstance(img, np.ndarray)
    assert img.shape == (256, 256, 3)

def test_save_image(tmp_path):
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    out_path = tmp_path / "out.png"
    save_image(img, out_path)
    assert out_path.exists()
    
    loaded = load_image(out_path)
    assert loaded.shape == (256, 256, 3)

def test_numpy_to_tensor():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tensor = numpy_to_tensor(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0

def test_tensor_to_numpy():
    tensor = torch.rand(1, 3, 256, 256)
    img = tensor_to_numpy(tensor)
    assert isinstance(img, np.ndarray)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8

def test_resize_for_inference():
    # Odd dimension
    img = np.random.randint(0, 255, (123, 125, 3), dtype=np.uint8)
    tensor = numpy_to_tensor(img)
    padded, original_size = resize_for_inference(tensor, 8)
    assert padded.shape[0] == 1
    assert padded.shape[2] % 8 == 0
    assert padded.shape[3] % 8 == 0
    assert original_size[0] == 123
    assert original_size[1] == 125

def test_unpad_inference():
    tensor = torch.rand(1, 3, 128, 128)
    unpadded = unpad_inference(tensor, (123, 125))
    assert unpadded.shape == (1, 3, 123, 125)
