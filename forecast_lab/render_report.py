"""Regenerate an offline report from existing measurements; never trains a model."""
import argparse
from pathlib import Path

from tslab.report import render_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    render_report(args.run_dir, export_png=args.png)
    print(args.run_dir / "report.html")
