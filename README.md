# Image Restoration AI System

An intelligent image restoration pipeline that **automatically detects and fixes** three types of visual degradation — **haze, low-light, and rain** — using state-of-the-art Transformer neural networks.

The system uses a trained lightweight MobileNetV2 CNN classifier (or classical CV detectors) to analyze an input image, determines which degradations are present with 99.9% accuracy, and routes it through the appropriate deep learning models — all without manual intervention.

---

## Features

| Feature | Description |
|---------|-------------|
| **Auto-Detection** | CNN Learned Detector (MobileNetV2) & Classical detectors (Dark Channel Prior, LAB luminance, Streak analysis) |
| **Smart Routing** | Mutual-exclusion router prevents false positives (e.g. vertical haze textures triggering deraining) |
| **SOTA Models** | DehazeFormer (IEEE TIP 2023), RetinexFormer (ICCV 2023), Restormer (CVPR 2022) |
| **Optimized Inference** | FP16 autocasting, `torch.compile`, lazy model loading, self-ensemble TTA |
| **CLI Interface** | Process single images or entire folders from the command line |
| **Quality Metrics** | Built-in PSNR, SSIM, LPIPS, and BRISQUE evaluation |
| **Benchmarking** | Evaluate model quality against ground-truth datasets |

---

## Benchmark Results (RTX 4050 Laptop GPU)

| Model | Dataset | Avg PSNR | Paper PSNR | Avg SSIM | Speed |
|-------|---------|----------|-----------|----------|-------|
| DehazeFormer-B | SOTS Indoor (100 images) | **38.31 dB** | 40.19 dB | 0.9906 | 0.84s/img |
| Restormer | Rain100L (100 images) | **37.47 dB** | 38.99 dB | 0.9740 | 1.07s/img |
| RetinexFormer (Single-Stage) | LOL-v2-Real (100 images) | **22.84 dB** | 27.18 dB | 0.8536 | 0.95s/img |

---

## Installation

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ (recommended) or CPU
- ~1 GB disk space for model weights

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/srihanrajguduru/Image_enhancement_project.git
cd Image_enhancement_project

# 2. Create a Conda environment
conda create --prefix ./env python=3.11 -y
conda activate ./env

# 3. Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements.txt
```

> **PowerShell users**: If `conda activate ./env` fails, use `.\env\python.exe` directly.

### Download Model Weights

Place pre-trained checkpoints in the `weights/` directory:

| Weight File | Model | Size | Download / Source |
|-------------|-------|------|-------------------|
| `detector.pth` | MobileNetV2 Degradation Classifier | 14 MB | Included / Trained |
| `dehazeformer-b.pth` | DehazeFormer-B | 11 MB | [DehazeFormer releases](https://github.com/IDKiro/DehazeFormer) |
| `LOL_v2_real.pth` | RetinexFormer | 6 MB | [RetinexFormer releases](https://github.com/caiyuanhao1998/Retinexformer) |
| `deraining.pth` | Restormer (deraining) | 100 MB | [Download](https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth) |
| `real_denoising.pth` | Restormer (denoising) | 100 MB | [Download](https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth) |

---

## Usage

### CLI — Single Image

```bash
# Basic restoration (auto-detects degradation)
python -m src.cli.restore images/input/hazy_photo.jpg --output output/restored.png

# With quality metrics against ground truth
python -m src.cli.restore images/input/hazy_photo.jpg --output output/restored.png \
  --reference images/reference/clean.png --metrics

# Force CPU
python -m src.cli.restore images/input/hazy_photo.jpg --output output/restored.png --device cpu
```

### CLI — Batch Processing

```bash
# Process all images in a folder
python -m src.cli.restore images/input/ --output output/
```

### Benchmarking

```bash
# Run benchmark with ground-truth references
python scripts/benchmark.py \
  --input images/haze_test/hazy/ \
  --reference images/haze_test/clear/ \
  --output output/benchmark_results.csv
```

### Python API

```python
from src.core.pipeline import RestorationPipeline

pipeline = RestorationPipeline()
result = pipeline.process_image(
    "images/input/hazy_photo.jpg",
    save_path="output/restored.png",
    reference_path="images/reference/clean.png"  # optional
)

print(f"Detected: {result['plan']}")       # ['haze']
print(f"PSNR: {result['metrics']['psnr']:.2f} dB")  # 38.31 dB
print(f"Time: {result['latency_ms']:.0f} ms")       # 840 ms
```

---

## Architecture

```
Input Image
    |
    v
[Degradation Detectors]  ── MobileNetV2 CNN Classifier (Default)
    |                     ── Heuristic Detectors (Fallback: DCP, LAB, Morphological)
    v
[Router]  ── Compares scores against thresholds
    |     ── Mutual-exclusion filtering (resolves haze vs rain conflicts)
    |     ── Builds execution plan: e.g., ["haze"]
    v
[Model Pipeline]  ── Sequentially applies needed models
    |             ── DehazeFormer (2.5M params, 9.6 MB)
    |             ── Restormer (26.1M params, 99.7 MB)
    |             ── RetinexFormer (1.6M params, 6.2 MB)
    v
Restored Image + Quality Metrics
```

---

## Configuration

### `configs/config.yaml`

```yaml
pipeline:
  device: "cuda"           # cuda, cpu, or mps
  precision: "fp32"        # fp32, fp16, or bf16
  compile_models: false    # Enable torch.compile
  self_ensemble: false     # 8x TTA (slower but +0.1-0.5 dB PSNR)
  lowlight_denoise: false  # Skip Restormer denoising after RetinexFormer (improves PSNR)

router:
  haze_threshold: 0.45     # Detection threshold for haze
  lowlight_threshold: 0.50  # Detection threshold for low-light
  rain_threshold: 0.45     # Detection threshold for rain
  execution_order: ["lowlight", "rain", "haze"]
  detector_type: "learned" # "learned" (MobileNetV2 CNN) or "heuristic" (classical CV)
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v
```

---

## License

This project uses publicly available pre-trained models. Please refer to the original repositories for their respective licenses.
