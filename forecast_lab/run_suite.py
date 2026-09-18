"""Sequential server experiments. Baseline first; regularization is opt-in."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from tslab.config import Config
from tslab.engine import train


DATASETS = {
    "ETTh1": "ETT-small/ETTh1.csv", "ETTh2": "ETT-small/ETTh2.csv",
    "ETTm1": "ETT-small/ETTm1.csv", "ETTm2": "ETT-small/ETTm2.csv",
    "weather": "weather/weather.csv", "electricity": "electricity/electricity.csv",
    "traffic": "traffic/traffic.csv", "exchange_rate": "exchange_rate/exchange_rate.csv",
    "illness": "illness/national_illness.csv",
}
VARIANT_KEYS = {"reconstruction_weight", "variance_weight", "covariance_weight"}
LOCATION_KEYS = {"output", "data", "device", "workers", "tensorboard"}


def assert_comparable(saved, requested, allow_regularization=False):
    saved = asdict(Config(**saved))  # Old checkpoints omit the new opt-in fields.
    ignored = LOCATION_KEYS | (VARIANT_KEYS if allow_regularization else set())
    differences = [key for key, value in asdict(requested).items()
                   if key not in ignored and saved.get(key) != value]
    if differences:
        raise ValueError(f"Existing run has different configuration: {differences}. Use a new output root")
    if Path(saved["data"]).resolve() != Path(requested.data).resolve():
        raise ValueError("Existing run uses a different CSV path; use a new output root")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument("--data", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=["ETTh1"])
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs/suite"))
    parser.add_argument("--stage", choices=("baseline", "regularized", "all"), default="baseline")
    parser.add_argument("--variants", nargs="+", choices=("variance", "varcov", "reconstruction", "combined"),
                        default=["variance", "varcov", "reconstruction"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[96])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--covariance-weight", type=float, default=0.001)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--skip-complete", action="store_true", help="Skip matching completed runs; never resume partial runs")
    args = parser.parse_args()
    base_values = json.loads(args.base_config.read_text(encoding="utf-8")) if args.base_config else {}
    base = Config(**base_values)
    for key in ("lookback", "epochs", "batch_size", "device", "workers"):
        value = getattr(args, key)
        if value is not None:
            setattr(base, key, value)
    variants = {
        "baseline": (0.0, 0.0, 0.0),
        "variance": (0.0, args.variance_weight, 0.0),
        "varcov": (0.0, args.variance_weight, args.covariance_weight),
        "reconstruction": (args.reconstruction_weight, 0.0, 0.0),
        "combined": (args.reconstruction_weight, args.variance_weight, args.covariance_weight),
    }
    requested = ["baseline"] if args.stage == "baseline" else list(dict.fromkeys(args.variants))
    if args.stage == "all":
        requested.insert(0, "baseline")
    datasets = {args.data.stem: args.data} if args.data else {name: args.data_root / DATASETS[name] for name in args.datasets}
    jobs = []
    # Validate all paths/configurations before any expensive training begins.
    for name, path in datasets.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        for horizon in dict.fromkeys(args.horizons):
            for seed in dict.fromkeys(args.seeds):
                parent = args.output_root / name / f"h{horizon}" / f"seed{seed}"
                common = replace(base, data=str(path.resolve()), horizon=horizon, seed=seed)
                if args.stage == "regularized":
                    reference = parent / "baseline" / "summary.json"
                    if not reference.is_file():
                        raise ValueError(f"Run and inspect baseline first: missing {reference}")
                    baseline_summary = json.loads(reference.read_text(encoding="utf-8"))
                    if baseline_summary.get("status") != "complete":
                        raise ValueError(f"Baseline has not completed: {reference}")
                    assert_comparable(baseline_summary["config"], common, allow_regularization=True)
                for variant in requested:
                    rec, var, cov = variants[variant]
                    cfg = replace(common, output=str(parent / variant), reconstruction_weight=rec,
                                  variance_weight=var, covariance_weight=cov).validate()
                    directory = Path(cfg.output)
                    if directory.exists() and any(directory.iterdir()):
                        summary_path = directory / "summary.json"
                        if args.skip_complete and summary_path.is_file():
                            summary = json.loads(summary_path.read_text(encoding="utf-8"))
                            if summary.get("status") != "complete":
                                raise ValueError(f"Run is not complete: {directory}")
                            assert_comparable(summary["config"], cfg)
                            print(f"SKIP completed: {directory}", flush=True)
                            continue
                        raise FileExistsError(f"Nonempty run directory: {directory}; choose another root or --skip-complete")
                    jobs.append(cfg)
    for index, cfg in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {cfg.output}", flush=True)
        train(cfg)
    print(f"Compare with: python compare_runs.py --root {args.output_root} --output {args.output_root}/comparison")


if __name__ == "__main__":
    main()
