"""Re-sample validation probes for a FROZEN checkpoint; never trains a model."""
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from tslab.config import Config
from tslab.data import load_data, probe_dataset
from tslab.diagnostics import run_diagnostics
from tslab.engine import export_evaluation, make_loader, select_device
from tslab.io import write_csv, write_json
from tslab.metrics import evaluate_forecasts
from tslab.model import ForecastEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", help="Relocated copy of the same CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sampling-seed", type=int, default=7301)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.repeats < 2 or args.windows < 2:
        raise ValueError("Use at least two repetitions and two windows")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Audit output must be new or empty")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg = Config(**checkpoint["config"])
    cfg.probe_windows = args.windows
    cfg.device = args.device
    if args.data:
        cfg.data = args.data
    cfg.validate()
    device = select_device(cfg.device)
    bundle = load_data(cfg, fitted=checkpoint["preprocessing"])
    model = ForecastEncoder(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, examples = evaluate_forecasts(model, make_loader(bundle.datasets["val"], cfg, device), bundle, cfg, device)
    export_evaluation(args.output / "full_validation", metrics, examples, bundle)
    reference = metrics["model"]["standardized"]["mse"]
    rows, intervention_rows, origin_sets = [], [], set()
    for repeat in range(args.repeats):
        seed = args.sampling_seed + repeat
        dataset = probe_dataset(bundle, cfg, "val", sampling_seed=seed)
        origin_sets.add(tuple(dataset.origins.tolist()))
        diagnostic = run_diagnostics(model, dataset, bundle, cfg, device, checkpoint["epoch"])
        diagnostic["sampling_seed"] = seed
        write_json(args.output / f"repeat_{repeat + 1:02d}.json", diagnostic)
        normal = diagnostic["ablations"]["normal"]["standardized"]["mse"]
        stats = diagnostic["statistics"]
        rows.append({"repeat": repeat + 1, "sampling_seed": seed, "probe_count": len(dataset),
                     "normal_mse": normal, "full_validation_mse": reference,
                     "probe_relative_bias": normal / reference - 1 if reference > 1e-12 else None,
                     "effective_rank": stats.get("effective_rank"), "mean_std": stats.get("mean_std")})
        for mode, result in diagnostic["ablations"].items():
            intervention_rows.append({"repeat": repeat + 1, "mode": mode, "mse": result["standardized"]["mse"],
                                      "delta_mse": result["delta_mse"],
                                      "mse_ratio": result["standardized"]["mse"] / normal if normal > 1e-12 else None})
        print(f"Repeat {repeat + 1}/{args.repeats}: {len(dataset)} probes, MSE={normal:.6f}", flush=True)
    aggregate = {}
    for key in ("normal_mse", "probe_relative_bias", "effective_rank", "mean_std"):
        values = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
        aggregate[key] = {"mean": float(values.mean()) if len(values) else None,
                          "std": float(values.std(ddof=1)) if len(values) > 1 else None}
    notes = ["One frozen checkpoint; variation is from probe sampling, NOT independent training seeds.",
             "Repeated subsets can overlap; error bars are descriptive, not confidence intervals.",
             "Each repetition uses the same intervention permutation algorithm/seed on a new chronological subset."]
    if len(origin_sets) < args.repeats:
        notes.append("Some/all origin sets repeat: requested window count can exhaust the separated candidate pool.")
    write_json(args.output / "audit_summary.json", {"checkpoint": str(args.run_dir / "best.pt"),
               "full_validation_mse": reference, "repeats": args.repeats, "unique_origin_sets": len(origin_sets),
               "aggregate": aggregate, "notes": notes, "config": vars(cfg)})
    write_csv(args.output / "probe_repeats.csv", rows)
    write_csv(args.output / "intervention_repeats.csv", intervention_rows)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Probe MSE vs full validation", "Intervention / normal MSE"))
    fig.add_trace(go.Scatter(x=[r["repeat"] for r in rows], y=[r["normal_mse"] for r in rows],
                             mode="lines+markers", name="Probe subsets"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r["repeat"] for r in rows], y=[reference] * len(rows),
                             mode="lines", line=dict(dash="dash"), name="Full validation"), row=1, col=1)
    modes = list(dict.fromkeys(r["mode"] for r in intervention_rows
                              if r["mode"] != "normal" and r["mse_ratio"] is not None))
    for mode in modes:
        fig.add_trace(go.Box(y=[r["mse_ratio"] for r in intervention_rows if r["mode"] == mode],
                             name=mode, boxpoints="all"), row=1, col=2)
    fig.update_layout(template="plotly_white", height=520, title="Fixed-model probe sampling audit")
    fig.update_xaxes(title="Repeat", row=1, col=1)
    content = fig.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True, "displaylogo": False})
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Probe Audit</title><body style="font:15px system-ui;margin:24px">'
    page += f'<h1>验证探针重复抽样</h1><p>完整验证 MSE：{reference:.6f}；不同窗口集合：{len(origin_sets)}/{args.repeats}。</p>'
    page += '<p>固定同一个模型。这里的波动来自探针抽样，不是随机种子训练稳定性；子集可能重叠，图中离散程度不代表置信区间。若窗口集合重复，降低 windows 或扩大可用数据后再分析抽样变异。</p>'
    page += content + '</body></html>'
    (args.output / "audit.html").write_text(page, encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot([r["repeat"] for r in rows], [r["normal_mse"] for r in rows], "o-", label="Probe MSE")
    axes[0].axhline(reference, linestyle="--", label="Full validation")
    axes[0].set(xlabel="Sampling repeat", ylabel="Standardized MSE")
    axes[0].legend()
    if modes:
        axes[1].boxplot([[r["mse_ratio"] for r in intervention_rows if r["mode"] == mode and r["mse_ratio"] is not None]
                         for mode in modes], labels=modes)
    else:
        axes[1].text(0.5, 0.5, "Ratios unavailable: normal MSE is zero", ha="center", transform=axes[1].transAxes)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(ylabel="Intervention / normal MSE")
    figure.tight_layout()
    figure.savefig(args.output / "audit.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output / "audit.html")


if __name__ == "__main__":
    main()
