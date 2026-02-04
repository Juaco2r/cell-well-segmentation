import argparse
from pathlib import Path
import glob

from .pipeline import run_batch

def parse_args():
    p = argparse.ArgumentParser(
        description="Cell well segmentation + feature extraction pipeline"
    )
    p.add_argument(
        "--input",
        required=True,
        help='Input TIFF(s). Examples: "data\\*.tif" or "C:\\path\\img.tif"',
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output root folder. If omitted, outputs are next to each input file.",
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Expand patterns like *.tif
    paths = glob.glob(args.input)

    if len(paths) == 0:
        raise SystemExit(f"No files matched input pattern: {args.input}")

    # For now, the pipeline writes outputs next to the input file.
    if args.out is not None:
        print("Note: --out is not implemented yet in process_single_file.")
        print("Outputs will still be saved next to each input file.")

    run_batch([Path(p) for p in paths])

if __name__ == "__main__":
    main()
