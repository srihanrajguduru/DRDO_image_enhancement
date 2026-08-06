# Complete CLI Commands Reference Guide

A comprehensive, all-in-one cheat sheet for running, testing, benchmarking, and processing images using the **Image Restoration AI System**.

---

## 1. Environment Setup & Activation

### PowerShell / Windows Terminal
```powershell
# Navigate to project directory
cd d:\image_drdo

# Activate local Conda environment
conda activate .\env

# OR run Python directly without activating Conda
.\env\python.exe <command>
```

---

## 2. Single Image Restoration Commands

Use `src.cli.restore` to restore single images with auto-detection or manual overrides.

### Basic Single Image Restoration
Auto-detects degradations (haze, low-light, rain) using the MobileNetV2 CNN classifier and saves the restored image:
```bash
python -m src.cli.restore images/haze_test/hazy/0001_0.8_0.2.jpg --output output/restored_0001.png
```

### With Quality Metrics & Ground-Truth Reference
Calculates **Full-Reference Metrics (PSNR, SSIM, LPIPS)** alongside **BRISQUE**:
```bash
python -m src.cli.restore images/haze_test/hazy/0001_0.8_0.2.jpg \
  --output output/restored_0001.png \
  --reference images/haze_test/clear/0001.png \
  --metrics
```

### Force CPU Execution (If GPU is unavailable)
```bash
python -m src.cli.restore images/lowlight_test/Low/low00690.png \
  --output output/restored_lowlight.png \
  --device cpu \
  --metrics
```

### Run Evaluation without Saving Restored Image (`--no-save`)
Useful for fast testing or metric inspection without disk writes:
```bash
python -m src.cli.restore images/rain_test/input/1.png \
  --reference images/rain_test/target/1.png \
  --metrics \
  --no-save
```

---

## 3. Batch Directory Restoration Commands

Process an entire directory of degraded images in one command.

### Process All Images in a Directory
Restores every image in `images/input/` and saves them to `output/restored_batch/`:
```bash
python -m src.cli.restore images/haze_test/hazy/ --output output/restored_haze_batch/
```

### Batch Processing with Ground-Truth Directory & Metrics
Matches input filenames against reference filenames in `images/haze_test/clear/` and reports metrics for each image:
```bash
python -m src.cli.restore images/haze_test/hazy/ \
  --output output/restored_haze_batch/ \
  --reference images/haze_test/clear/ \
  --metrics
```

---

## 4. Full Dataset Benchmarking & Average Metrics

Run the automated benchmarking script across test datasets to process **all images** and generate **average metrics (PSNR, SSIM, LPIPS, BRISQUE, Latency)** into CSV and Markdown summary reports.

### Benchmark Haze Dataset (SOTS Indoor)
```bash
python scripts/benchmark.py \
  --input images/haze_test/hazy/ \
  --reference images/haze_test/clear/ \
  --output output/haze_benchmark.csv \
  --device cuda
```
*Generates `output/haze_benchmark.csv` and summary `output/haze_benchmark.md`.*

### Benchmark Rain Dataset (Rain100L)
```bash
python scripts/benchmark.py \
  --input images/rain_test/input/ \
  --reference images/rain_test/target/ \
  --output output/rain_benchmark.csv \
  --device cuda
```

### Benchmark Low-Light Dataset (LOL-v2-Real)
```bash
python scripts/benchmark.py \
  --input images/lowlight_test/Low/ \
  --reference images/lowlight_test/Normal/ \
  --output output/lowlight_benchmark.csv \
  --device cuda
```

### No-Reference Dataset Benchmark (Without Ground-Truth)
If test images do not have matching reference clean images:
```bash
python scripts/benchmark.py \
  --input images/input/ \
  --output output/unlabeled_benchmark.csv \
  --device cuda
```

---

## 5. System Testing Commands

Run unit and integration test suites to verify system health.

### Run All Unit & Integration Tests (PyTest)
```bash
python -m pytest tests/ -v
```

### Run Only Unit Tests
```bash
python -m pytest tests/unit/ -v
```

### Run Only Integration Tests
```bash
python -m pytest tests/integration/ -v
```

---

## 6. Python API Snippet Reference

You can also call the pipeline directly inside any Python script:

```python
from src.core.pipeline import RestorationPipeline

# 1. Initialize Pipeline
pipeline = RestorationPipeline()

# 2. Process Image
result = pipeline.process_image(
    "images/haze_test/hazy/0001_0.8_0.2.jpg",
    save_path="output/restored.png",
    reference_path="images/haze_test/clear/0001.png" # Optional
)

# 3. Access Output Attributes
print("Degradation Plan:", result["plan"])          # e.g. ['haze']
print("Detector Scores:", result["scores"])         # {'haze': 0.99, 'lowlight': 0.00, 'rain': 0.00}
print("Latency:", f"{result['latency_ms']:.2f} ms") # e.g. 840.12 ms
print("PSNR:", result["metrics"].get("psnr"))       # 38.31 dB
print("SSIM:", result["metrics"].get("ssim"))       # 0.9906
```

---

## Summary Table of Arguments (`src.cli.restore`)

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `input` | — | `str` | Required | Path to input image file or directory |
| `--output` | `-o` | `str` | `None` | Path to save restored image file or output directory |
| `--reference` | `-r` | `str` | `None` | Ground-truth reference image file or directory |
| `--device` | `-d` | `str` | `cuda` | Hardware execution target (`cuda`, `cpu`, `mps`) |
| `--metrics` | `-m` | `flag` | `False` | Calculate and print PSNR, SSIM, LPIPS, BRISQUE |
| `--no-save` | — | `flag` | `False` | Skip saving output image to disk |
| `--config` | `-c` | `str` | `None` | Path to custom YAML configuration file |
