import argparse
import sys
from pathlib import Path
import json

from src.core.pipeline import RestorationPipeline
from src.utils.logger import get_logger

logger = get_logger("cli")


def main():
    parser = argparse.ArgumentParser(description="Image Restoration CLI")
    parser.add_argument("input", type=str, help="Path to input image or directory")
    parser.add_argument(
        "--output", type=str, default="output", help="Path to output image or directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        help="Override device configuration",
    )
    parser.add_argument("--metrics", action="store_true", help="Print quality metrics")
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Path to clean ground-truth image for full-reference metrics (PSNR/SSIM/LPIPS)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Update global config before initializing pipeline
    from src.core.config import get_config
    if args.device:
        get_config().pipeline.device = args.device

    # Initialize pipeline
    pipeline = RestorationPipeline()

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Process single file
    if input_path.is_file():
        if not output_path.suffix:
            output_file = output_path / f"restored_{input_path.name}"
        else:
            output_file = output_path

        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            results = pipeline.process_image(input_path, output_file, reference_path=args.reference)
            if args.metrics:
                print(json.dumps(results, indent=2))
            else:
                print(f"Successfully restored {input_path} -> {output_file}")
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {e}")
            sys.exit(1)

    # Process directory
    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        success_count = 0
        total_count = 0

        for file in input_path.iterdir():
            if file.suffix.lower() in image_extensions:
                total_count += 1
                out_file = output_path / f"restored_{file.name}"
                try:
                    results = pipeline.process_image(file, out_file, reference_path=args.reference)
                    if args.metrics:
                        print(f"{file.name}: {results['metrics']}")
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to process {file}: {e}")

        print(
            f"Successfully processed {success_count}/{total_count} images in {input_path}"
        )


if __name__ == "__main__":
    main()
