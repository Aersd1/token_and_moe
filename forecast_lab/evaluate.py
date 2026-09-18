"""Re-evaluate a selected checkpoint on its original chronological split."""
import argparse
import json
from pathlib import Path

import torch

from tslab.config import Config
from tslab.data import load_data, probe_dataset
from tslab.diagnostics import run_diagnostics
from tslab.engine import export_evaluation, make_loader, provenance, seed_everything, select_device
from tslab.io import write_json
from tslab.metrics import evaluate_forecasts
from tslab.model import ForecastEncoder
from tslab.report import render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", help="Relocated copy of the SAME CSV (schema and boundaries must match)")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", action="store_true", help="Optional frozen probes on the selected split")
    parser.add_argument("--output-constraint", choices=("none", "nonnegative", "bounded"),
                        help="Evaluate extra bounded outputs, without changing the checkpoint")
    parser.add_argument("--output-min", type=float)
    parser.add_argument("--output-max", type=float)
    parser.add_argument("--probe-windows", type=int)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Evaluation output must be new or empty")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg = Config(**checkpoint["config"])
    if args.data:
        cfg.data = args.data
    if args.batch_size:
        cfg.batch_size = args.batch_size
    cfg.device = args.device
    for name in ("output_constraint", "output_min", "output_max", "probe_windows"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    cfg.validate()
    seed_everything(cfg.seed, cfg.deterministic)
    device = select_device(cfg.device)
    bundle = load_data(cfg, fitted=checkpoint["preprocessing"])
    model = ForecastEncoder(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = make_loader(bundle.datasets[args.split], cfg, device)
    metrics, examples = evaluate_forecasts(model, loader, bundle, cfg, device)
    args.output.mkdir(parents=True, exist_ok=True)
    export_evaluation(args.output, metrics, examples, bundle)
    diagnostic = None
    if args.diagnostics:
        diagnostic = run_diagnostics(model, probe_dataset(bundle, cfg, args.split), bundle, cfg,
                                     device, checkpoint["epoch"], split=args.split)
        write_json(args.output / "diagnostic.json", diagnostic)
    state = {"config": vars(cfg), "data": bundle.metadata, "status": "evaluation complete",
             "provenance": provenance(device), "best_val_mse": checkpoint["val_mse"],
             "diagnostic": diagnostic,
             "evaluation": {"split": args.split, "metrics": metrics, "examples": examples},
             "history": [], "monitor_history": []}
    write_json(args.output / "report_state.json", state)
    render_report(args.output, export_png=True)
    print(json.dumps(metrics["model"]["standardized"], indent=2))
    print(f"Report: {args.output / 'report.html'}")


if __name__ == "__main__":
    main()
