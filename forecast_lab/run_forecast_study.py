"""Controlled direct/residual x uniform/near-weighted experiments and width sweeps."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from run_suite import assert_comparable
from tslab.config import Config
from tslab.engine import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs/forecast_study"))
    parser.add_argument("--variants", nargs="+", choices=("baseline", "residual", "near", "residual_near"),
                        default=["baseline", "residual", "near", "residual_near"])
    parser.add_argument("--widths", nargs="+", type=int, default=[64])
    parser.add_argument("--horizons", nargs="+", type=int, default=[24, 96, 244])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--near-weight", type=float, default=4.0)
    parser.add_argument("--near-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without training or writing runs")
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.near_weight <= 1:
        raise ValueError("For this ablation, --near-weight must exceed 1; baseline/residual use 1")
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    base = Config(**values)
    for key in ("near_steps", "epochs", "device"):
        if getattr(args, key) is not None:
            setattr(base, key, getattr(args, key))
    jobs = []
    source_hash = None
    for horizon in dict.fromkeys(args.horizons):
        for width in dict.fromkeys(args.widths):
            for seed in dict.fromkeys(args.seeds):
                for variant in dict.fromkeys(args.variants):
                    output = args.output_root / args.data.stem / f"h{horizon}" / f"d{width}" / f"seed{seed}" / variant
                    cfg = replace(base, data=str(args.data.resolve()), output=str(output), horizon=horizon,
                                  d_model=width, seed=seed,
                                  prediction_mode="residual" if variant in {"residual", "residual_near"} else "direct",
                                  near_weight=args.near_weight if variant in {"near", "residual_near"} else 1.0,
                                  reconstruction_weight=0.0, variance_weight=0.0, covariance_weight=0.0).validate()
                    if not args.dry_run and output.exists() and any(output.iterdir()):
                        summary_file = output / "summary.json"
                        if not args.skip_complete or not summary_file.is_file():
                            raise FileExistsError(f"Nonempty output: {output}. Choose a new root or --skip-complete")
                        summary = json.loads(summary_file.read_text(encoding="utf-8"))
                        if summary.get("status") != "complete":
                            raise ValueError(f"Incomplete run: {output}")
                        assert_comparable(summary["config"], cfg)
                        if source_hash is None:
                            digest = hashlib.sha256()
                            with args.data.open("rb") as handle:
                                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                                    digest.update(chunk)
                            source_hash = digest.hexdigest()
                        if summary.get("data_sha256") != source_hash:
                            raise ValueError(f"Existing run data fingerprint missing or changed: {output}")
                        print(f"SKIP matching completed run: {output}", flush=True)
                        continue
                    jobs.append(cfg)
    print(f"Planned runs: {len(jobs)}. Checkpoint selection: raw unweighted validation MSE.", flush=True)
    for index, cfg in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {cfg.output} | mode={cfg.prediction_mode} near_weight={cfg.near_weight}", flush=True)
        if not args.dry_run:
            train(cfg)
    print(f"Compare: python compare_runs.py --study --root {args.output_root} --output {args.output_root}/comparison")


if __name__ == "__main__":
    main()
