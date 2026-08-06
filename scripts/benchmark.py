import argparse
import time
import pandas as pd
import torch
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.pipeline import RestorationPipeline
from src.utils.image import load_image

def run_benchmark(input_dir: str, output_csv: str, device: str = None, reference_dir: str = None):
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: {input_dir} is not a valid directory.")
        return

    from src.core.config import get_config
    if device:
        get_config().pipeline.device = device

    pipeline = RestorationPipeline()
    results = []

    image_extensions = {".jpg", ".jpeg", ".png"}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(images)} images to benchmark.")

    reference_path = Path(reference_dir) if reference_dir else None

    for img_path in tqdm(images, desc="Benchmarking"):
        try:
            ref_img_path = None
            if reference_path and reference_path.exists():
                potential_ref = reference_path / img_path.name
                if potential_ref.exists():
                    ref_img_path = str(potential_ref)
                    
            # We don't save the image during benchmarking to isolate inference time
            res = pipeline.process_image(img_path, reference_path=ref_img_path)
            
            row = {
                "filename": img_path.name,
                "latency_ms": res["latency_ms"],
                "plan": "->".join(res["plan"]) if res["plan"] else "None",
            }
            
            # Add scores
            for k, v in res["scores"].items():
                row[f"score_{k}"] = round(v, 4)
                
            # Add metrics
            for k, v in res["metrics"].items():
                row[f"metric_{k}"] = round(v, 4) if v is not None else None
                
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Generate CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Benchmark complete. Results saved to {output_csv}")

    # Generate Markdown Report
    md_path = output_csv.replace(".csv", ".md")
    with open(md_path, "w") as f:
        f.write("# Benchmark Report\n\n")
        f.write("## Summary\n")
        f.write(f"- Total Images Processed: {len(df)}\n")
        f.write(f"- Average Latency: {df['latency_ms'].mean():.2f} ms\n")
        
        if "metric_psnr" in df.columns:
            f.write(f"- Average PSNR: {df['metric_psnr'].mean():.2f} dB\n")
            f.write(f"- Average SSIM: {df['metric_ssim'].mean():.4f}\n")
            f.write(f"- Average LPIPS: {df['metric_lpips'].mean():.4f}\n")
            f.write(f"- Average BRISQUE: {df['metric_brisque'].mean():.2f}\n")
            
        f.write("\n## Details\n")
        f.write(df.to_markdown(index=False))
        
    print(f"Markdown report generated at {md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Image Restoration Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV file path")
    parser.add_argument("--reference", type=str, default=None, help="Directory containing ground-truth reference images (for PSNR/SSIM/LPIPS)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None, help="Device to run on")
    
    args = parser.parse_args()
    run_benchmark(args.input, args.output, args.device, args.reference)
