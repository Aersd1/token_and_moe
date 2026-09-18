"""Compare completed runs without reloading datasets or training models."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from tslab.io import write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    config_keys_ignored = {"output", "seed", "device", "workers", "tensorboard",
                          "reconstruction_weight", "variance_weight", "covariance_weight"}
    for path in sorted(args.root.rglob("summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "complete":
            continue
        cfg = item["config"]
        protocol = {k: v for k, v in cfg.items() if k not in config_keys_ignored}
        protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
        rec, var, cov = (cfg[k] for k in ("reconstruction_weight", "variance_weight", "covariance_weight"))
        variant = "baseline" if not any((rec, var, cov)) else f"rec={rec:g}, var={var:g}, cov={cov:g}"
        test = item["test"]["model"]["standardized"]
        diagnostic = item.get("diagnostic", {})
        rows.append({"run": str(path.parent.resolve()), "dataset": Path(cfg["data"]).stem,
                     "protocol": protocol_id, "horizon": cfg["horizon"], "lookback": cfg["lookback"],
                     "seed": cfg["seed"], "variant": variant, "best_epoch": item["best_epoch"],
                     "val_mse": item["best_val_mse"], "test_mse": test["mse"], "test_mae": test["mae"],
                     "effective_rank": diagnostic.get("effective_rank"),
                     "mean_std": diagnostic.get("mean_std"),
                     "sample_shuffle_delta": diagnostic.get("sample_shuffle_delta_mse"),
                     "zero_delta": diagnostic.get("zero_delta_mse")})
    if not rows:
        raise ValueError("No completed summary.json files found")
    frame = pd.DataFrame(rows)
    # A duplicate seed would falsely look like another independent repetition.
    duplicate = frame.duplicated(["protocol", "variant", "seed"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate protocol/variant/seed runs. Narrow --root before averaging: "
                         + str(frame.loc[duplicate, "run"].tolist()))
    groups = []
    paired = []
    for (protocol, variant), group in frame.groupby(["protocol", "variant"], sort=True):
        first = group.iloc[0]
        record = {"protocol": protocol, "dataset": first["dataset"], "horizon": int(first["horizon"]),
                  "lookback": int(first["lookback"]), "variant": variant,
                  "n_seeds": len(group), "seeds": ",".join(map(str, sorted(group["seed"].tolist())))}
        for key in ("val_mse", "test_mse", "test_mae", "effective_rank", "mean_std", "sample_shuffle_delta"):
            numeric = pd.to_numeric(group[key], errors="coerce")
            record[f"{key}_mean"] = numeric.mean()
            record[f"{key}_std"] = numeric.std(ddof=1) if numeric.notna().sum() > 1 else np.nan
        groups.append(record)
        if variant != "baseline":
            reference = frame[(frame["protocol"] == protocol) & (frame["variant"] == "baseline")]
            joined = group.merge(reference, on="seed", suffixes=("_variant", "_baseline"))
            for _, row in joined.iterrows():
                paired.append({"protocol": protocol, "dataset": first["dataset"], "variant": variant,
                               "seed": int(row["seed"]), "horizon": int(first["horizon"]),
                               "delta_val_mse": row["val_mse_variant"] - row["val_mse_baseline"],
                               "delta_test_mse": row["test_mse_variant"] - row["test_mse_baseline"]})
    grouped = pd.DataFrame(groups).sort_values(["protocol", "val_mse_mean"])
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "grouped.csv", grouped.to_dict("records"))
    write_csv(args.output / "paired_deltas.csv", paired)
    write_json(args.output / "comparison.json", {"runs": rows, "groups": grouped.to_dict("records"), "paired": paired})
    figures = []
    for protocol, group in grouped.groupby("protocol"):
        title = f"{group.iloc[0]['dataset']} / H={group.iloc[0]['horizon']} / protocol {protocol}"
        for metric, label in (("val_mse", "Validation MSE (selection)"), ("test_mse", "Test MSE (report only)")):
            plot = px.bar(group, x="variant", y=f"{metric}_mean", error_y=f"{metric}_std", title=f"{title} · {label}",
                          color="variant", template="plotly_white", hover_data=["n_seeds", "seeds"])
            plot.update_layout(showlegend=False, xaxis_title="Variant", yaxis_title="Standardized MSE")
            figures.append(plot)
    page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Forecast Lab / Comparison</title>'
    page += '<style>body{font:15px system-ui;margin:32px;color:#23334c;background:#f5f7fb}table{border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #ddd}section{overflow:auto;margin:20px 0}h1{color:#10213d}</style></head><body>'
    page += '<h1>Forecast Lab / 实验对照</h1><p>每个 protocol 固定数据路径、划分、模型及训练配置。仅种子和辅助正则权重允许变化。误差条为种子间样本标准差；单种子没有误差条。不同种子集合不直接视为配对实验，配对差值见 paired_deltas.csv。</p>'
    page += '<p>按验证误差排序。测试指标只用于最终报告，不用于挑选正则权重。不同协议、量纲或预测长度之间不直接平均。</p>'
    for index, figure in enumerate(figures):
        page += figure.to_html(full_html=False, include_plotlyjs=True if index == 0 else False,
                               config={"displaylogo": False, "responsive": True})
    page += '<section>' + grouped.to_html(index=False, escape=True, float_format=lambda x: f"{x:.6g}") + '</section>'
    page += '<p>原始运行数：' + html.escape(str(len(rows))) + '</p></body></html>'
    (args.output / "comparison.html").write_text(page, encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for protocol, group in grouped.groupby("protocol"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric in zip(axes, ("val_mse", "test_mse")):
            ax.bar(group["variant"], group[f"{metric}_mean"], yerr=group[f"{metric}_std"].fillna(0), capsize=4)
            ax.tick_params(axis="x", rotation=25, labelsize=8)
            ax.set(title=metric, ylabel="Standardized MSE")
        fig.suptitle(f"{group.iloc[0]['dataset']} / H={group.iloc[0]['horizon']} / {protocol}")
        fig.tight_layout()
        fig.savefig(args.output / f"comparison_{protocol}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(args.output / "comparison.html")


if __name__ == "__main__":
    main()
