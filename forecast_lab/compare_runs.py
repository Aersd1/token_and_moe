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
from tslab.config import Config
from dataclasses import asdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", action="store_true", help="Compare prediction modes, near weights and widths as explicit factors")
    parser.add_argument("--reference-width", type=int, default=64)
    args = parser.parse_args()
    rows = []
    config_keys_ignored = {"output", "seed", "device", "workers", "tensorboard",
                          "reconstruction_weight", "variance_weight", "covariance_weight"}
    if args.study:
        config_keys_ignored |= {"d_model", "prediction_mode", "near_weight"}
    for path in sorted(args.root.rglob("summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "complete":
            continue
        cfg = asdict(Config(**item["config"]))
        protocol = {k: v for k, v in cfg.items() if k not in config_keys_ignored}
        protocol["data_sha256"] = item.get("data_sha256", "not_recorded")
        protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
        rec, var, cov = (cfg[k] for k in ("reconstruction_weight", "variance_weight", "covariance_weight"))
        variant = "baseline" if not any((rec, var, cov)) else f"rec={rec:g}, var={var:g}, cov={cov:g}"
        is_baseline = (not any((rec, var, cov)) and cfg["prediction_mode"] == "direct" and cfg["near_weight"] == 1.0)
        if args.study:
            variant = f"{cfg['prediction_mode']} / near={cfg['near_weight']:g} / D={cfg['d_model']} / {variant}"
        test = item["test"]["model"]["standardized"]
        bounded = item["test"].get("model_bounded", {}).get("standardized", {})
        audit = item["test"]["model"].get("range_audit", {})
        diagnostic = item.get("diagnostic", {})
        rows.append({"run": str(path.parent.resolve()), "dataset": Path(cfg["data"]).stem,
                     "data_sha256": item.get("data_sha256"),
                     "protocol": protocol_id, "horizon": cfg["horizon"], "lookback": cfg["lookback"],
                     "seed": cfg["seed"], "variant": variant, "best_epoch": item["best_epoch"],
                     "d_model": cfg["d_model"], "is_baseline": is_baseline,
                     "val_mse": item["best_val_mse"], "test_mse": test["mse"], "test_mae": test["mae"],
                     "effective_rank": diagnostic.get("effective_rank"),
                     "mean_std": diagnostic.get("mean_std"),
                     "sample_shuffle_delta": diagnostic.get("sample_shuffle_delta_mse"),
                     "zero_delta": diagnostic.get("zero_delta_mse")})
        rows[-1].update({"first_step_mse": test.get("first_step_mse"), "near_mse": test.get("near_mse"),
                         "far_mse": test.get("far_mse"), "bounded_mse": bounded.get("mse"),
                         "below_fraction": audit.get("below_fraction"), "above_fraction": audit.get("above_fraction"),
                         "parameters": item.get("parameter_count"), "training_seconds": item.get("training_seconds")})
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
                  "d_model": int(first["d_model"]),
                  "n_seeds": len(group), "seeds": ",".join(map(str, sorted(group["seed"].tolist())))}
        for key in ("val_mse", "test_mse", "test_mae", "effective_rank", "mean_std", "sample_shuffle_delta",
                    "first_step_mse", "near_mse", "far_mse", "bounded_mse", "below_fraction", "above_fraction",
                    "parameters", "training_seconds"):
            numeric = pd.to_numeric(group[key], errors="coerce")
            record[f"{key}_mean"] = numeric.mean()
            record[f"{key}_std"] = numeric.std(ddof=1) if numeric.notna().sum() > 1 else np.nan
        groups.append(record)
        width_reference = args.study and bool(first["is_baseline"]) and int(first["d_model"]) != args.reference_width
        if not bool(first["is_baseline"]) or width_reference:
            reference_width = args.reference_width if width_reference else int(first["d_model"])
            reference = frame[(frame["protocol"] == protocol) & frame["is_baseline"]
                              & (frame["d_model"] == reference_width)]
            joined = group.merge(reference, on="seed", suffixes=("_variant", "_baseline"))
            for _, row in joined.iterrows():
                paired.append({"protocol": protocol, "dataset": first["dataset"], "variant": variant,
                               "seed": int(row["seed"]), "horizon": int(first["horizon"]),
                               "comparison_type": "width_vs_reference" if width_reference else "forecast_change_same_width",
                               "reference_variant": row["variant_baseline"],
                               "delta_val_mse": row["val_mse_variant"] - row["val_mse_baseline"],
                               "delta_test_mse": row["test_mse_variant"] - row["test_mse_baseline"]})
                for metric in ("first_step_mse", "near_mse", "far_mse", "bounded_mse"):
                    value, baseline = row[f"{metric}_variant"], row[f"{metric}_baseline"]
                    paired[-1][f"delta_{metric}"] = value - baseline if pd.notna(value) and pd.notna(baseline) else None
    grouped = pd.DataFrame(groups).sort_values(["protocol", "val_mse_mean"])
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "grouped.csv", grouped.to_dict("records"))
    write_csv(args.output / "paired_deltas.csv", paired)
    write_json(args.output / "comparison.json", {"runs": rows, "groups": grouped.to_dict("records"), "paired": paired})
    figures = []
    for protocol, group in grouped.groupby("protocol"):
        title = f"{group.iloc[0]['dataset']} / H={group.iloc[0]['horizon']} / protocol {protocol}"
        for metric, label in (("val_mse", "Raw validation MSE (selection)"), ("test_mse", "Raw test MSE"),
                               ("first_step_mse", "First-step test MSE"), ("near_mse", "Near test MSE"),
                               ("bounded_mse", "Bounded test MSE (paired postprocessing)")):
            if group[f"{metric}_mean"].isna().all():
                continue
            plot = px.bar(group, x="variant", y=f"{metric}_mean", error_y=f"{metric}_std", title=f"{title} · {label}",
                          color="variant", template="plotly_white", hover_data=["n_seeds", "seeds"])
            plot.update_layout(showlegend=False, xaxis_title="Variant", yaxis_title="Standardized MSE")
            figures.append(plot)
        if group["parameters_mean"].notna().any():
            plot = px.scatter(group, x="parameters_mean", y="val_mse_mean", color="variant", size="d_model",
                              hover_data=["n_seeds", "training_seconds_mean"],
                              title=f"{title} · Capacity vs raw validation MSE", template="plotly_white")
            figures.append(plot)
    page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Forecast Lab / Comparison</title>'
    page += '<style>body{font:15px system-ui;margin:32px;color:#23334c;background:#f5f7fb}table{border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #ddd}section{overflow:auto;margin:20px 0}h1{color:#10213d}</style></head><body>'
    page += '<h1>Forecast Lab / 实验对照</h1><p>每个 protocol 固定数据版本、划分与共同训练条件。普通模式比较辅助正则；study 模式额外把编码宽度、直接/残差预测、近端权重作为明确实验因素。误差条为种子间样本标准差，单种子没有误差条。</p>'
    page += '<p>模式/损失对照匹配相同编码宽度与种子的直接预测基线；纯宽度对照匹配参考宽度（默认 64）的基线。缺少匹配运行时不伪造配对差值，详见 paired_deltas.csv。model_bounded 是输出限制后的结果，原始模型仍按未加权验证 MSE 选取。</p>'
    page += '<p>按验证误差排序。测试指标只用于最终报告，不用于挑选正则权重。不同协议、量纲或预测长度之间不直接平均。</p>'
    if any(row["data_sha256"] is None for row in rows):
        page += '<p>部分旧运行没有数据指纹，其分组仅基于已保存配置，不能证明数据内容完全相同；与含指纹的新运行分开比较。</p>'
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
