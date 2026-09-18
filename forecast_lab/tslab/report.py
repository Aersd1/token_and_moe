"""Offline interactive HTML and publication-friendly PNGs from recorded results.

No illustrative/fake measurements are generated. Empty sections stay empty until
the server has produced the corresponding result.
"""
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = ["#2764e7", "#ed8936", "#0d9488", "#a855f7", "#e04f75"]


def _figure(title, y_title=None):
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white", height=360,
                      margin=dict(l=55, r=20, t=60, b=48),
                      colorway=COLORS, font=dict(family="Arial, sans-serif", color="#334155"),
                      legend=dict(orientation="h", y=-0.2), hovermode="closest")
    if y_title:
        fig.update_yaxes(title=y_title)
    return fig


def make_figures(state):
    figures = []
    history = state.get("history", [])
    if history:
        fig = _figure("01 / Forecast learning curves", "Standardized MSE")
        for key, label in (("train_prediction", "Train (dropout on)"), ("val_mse", "Validation")):
            fig.add_trace(go.Scatter(x=[r["epoch"] for r in history], y=[r[key] for r in history],
                                     mode="lines+markers", name=label))
        fig.update_xaxes(title="Epoch")
        figures.append(("learning_curves", fig))
        if "val_near_mse" in history[0]:
            fig = _figure("Near / far validation errors (raw forecasts)", "Standardized MSE")
            for key, name in (("val_first_step_mse", "First step"), ("val_near_mse", "Near horizon"),
                              ("val_far_mse", "Far horizon")):
                fig.add_trace(go.Scatter(x=[r["epoch"] for r in history], y=[r.get(key) for r in history], name=name))
            figures.append(("near_far_learning", fig))
        auxiliary = _figure("02 / Auxiliary objectives (unweighted)", "Loss")
        for key in ("train_reconstruction", "train_variance", "train_covariance"):
            if any(r.get(key, 0) for r in history):
                auxiliary.add_trace(go.Scatter(x=[r["epoch"] for r in history],
                                               y=[r.get(key, 0) for r in history], name=key))
        if auxiliary.data:
            figures.append(("auxiliary_losses", auxiliary))
    monitor_history = state.get("monitor_history", [])
    if monitor_history:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Matched-position effective rank", "Cross-window std"))
        x = [r["epoch"] for r in monitor_history]
        for key, label, col in (("effective_rank", "Matched-position rank", 1),
                                ("pooled_effective_rank", "Pooled rank", 1),
                                ("mean_std", "Mean std", 2)):
            fig.add_trace(go.Scatter(x=x, y=[r.get(key) for r in monitor_history],
                                     name=label, mode="lines+markers"), row=1, col=col)
        fig.update_layout(title="03 / Representation trajectory (validation probes)", template="plotly_white",
                          height=360, colorway=COLORS, margin=dict(l=45, r=20, t=80, b=70),
                          legend=dict(orientation="h", y=-0.25))
        figures.append(("representation_trajectory", fig))
    diag = state.get("diagnostic") or {}
    stats = diag.get("statistics", {})
    label = f"{diag.get('split', 'val')} / epoch {diag.get('epoch', '?')}"
    if stats.get("available"):
        fig = _figure(f"04 / Feature activity · {label}", "Cross-window standard deviation")
        fig.add_trace(go.Bar(x=list(range(len(stats["std_per_dimension"]))), y=stats["std_per_dimension"],
                             marker_color=COLORS[0], name="Std"))
        fig.add_hline(y=stats["near_zero_threshold"], line_dash="dash", line_color=COLORS[4])
        fig.update_xaxes(title="Latent feature index")
        figures.append(("feature_activity", fig))
        fig = _figure(f"05 / Covariance spectrum · {label}", "Eigenvalue")
        for key, name in (("eigenvalues", "Matched-position"), ("pooled_eigenvalues", "Temporal mean")):
            fig.add_trace(go.Scatter(x=list(range(1, len(stats[key]) + 1)),
                                     y=np.maximum(stats[key], 1e-12), name=name, mode="lines+markers"))
        fig.update_yaxes(type="log")
        fig.update_xaxes(title="Eigenvalue rank")
        figures.append(("spectrum", fig))
        fig = _figure(f"06 / Feature correlation · {label}")
        fig.add_trace(go.Heatmap(z=stats["correlation"], zmin=-1, zmax=1, colorscale="RdBu",
                                 colorbar=dict(title="Correlation")))
        fig.update_xaxes(title="Feature")
        fig.update_yaxes(title="Feature")
        figures.append(("correlation", fig))
        fig = _figure(f"07 / Window representation PCA · {label}", "PC 2")
        coordinates = np.asarray(stats["pooled_pca"])
        fig.add_trace(go.Scatter(x=coordinates[:, 0], y=coordinates[:, 1], mode="markers",
                                 marker=dict(color=diag["origins"], colorscale="Viridis", size=8,
                                             colorbar=dict(title="Origin row")),
                                 text=[f"Origin row {x}" for x in diag["origins"]], name="Window mean"))
        fig.update_xaxes(title="PC 1")
        figures.append(("pca", fig))
        fig = _figure(f"08 / Window-mean cosine similarity · {label}")
        fig.add_trace(go.Heatmap(z=stats["pooled_cosine"], zmin=-1, zmax=1, colorscale="RdBu"))
        fig.update_xaxes(title="Probe window (chronological)")
        fig.update_yaxes(title="Probe window (chronological)")
        figures.append(("sample_similarity", fig))
        if "centered_pooled_cosine" in stats:
            fig = _figure(f"Centered window-mean cosine · {label}")
            fig.add_trace(go.Heatmap(z=stats["centered_pooled_cosine"], zmin=-1, zmax=1, colorscale="RdBu"))
            figures.append(("centered_similarity", fig))
    ablations = diag.get("ablations", {})
    if ablations:
        fig = _figure(f"09 / Latent interventions · {label}", "Standardized MSE (lower is better)")
        names = list(ablations)
        fig.add_trace(go.Bar(x=names, y=[ablations[k]["standardized"]["mse"] for k in names],
                             marker_color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                             customdata=[ablations[k]["delta_mse"] for k in names],
                             hovertemplate="%{x}<br>MSE=%{y:.6f}<br>ΔMSE=%{customdata:.6f}<extra></extra>"))
        figures.append(("interventions", fig))
        if all("bounded_metrics" in value for value in ablations.values()):
            fig = _figure(f"Paired raw / bounded interventions · {label}", "Standardized MSE")
            fig.add_trace(go.Bar(x=names, y=[ablations[k]["standardized"]["mse"] for k in names], name="Raw"))
            fig.add_trace(go.Bar(x=names, y=[ablations[k]["bounded_metrics"]["standardized"]["mse"] for k in names], name="Bounded"))
            figures.append(("bounded_interventions", fig))
    evaluation = state.get("evaluation") or {}
    metrics = evaluation.get("metrics", {})
    split = evaluation.get("split", "test")
    if metrics:
        fig = _figure(f"10 / Full {split} split · model vs naive forecasts", "Standardized MSE")
        names = list(metrics)
        fig.add_trace(go.Bar(x=names, y=[metrics[k]["standardized"]["mse"] for k in names],
                             marker_color=COLORS[:len(names)]))
        figures.append(("forecast_baselines", fig))
        fig = _figure(f"11 / Error by forecast lead · full {split}", "Standardized MSE")
        for name, result in metrics.items():
            values = result["standardized"]["horizon_mse"]
            fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)), y=values, name=name))
        fig.update_xaxes(title="Forecast lead (rows)")
        figures.append(("horizon_error", fig))
        if "first_step_mse" in metrics["model"]["standardized"]:
            fig = _figure(f"Near / far forecast quality · full {split}", "Standardized MSE")
            for name, result in metrics.items():
                values = result["standardized"]
                fig.add_trace(go.Bar(x=["First step", f"First {result['near_steps']} steps", "Remaining steps"],
                                     y=[values["first_step_mse"], values["near_mse"], values["far_mse"]], name=name))
            figures.append(("near_far_quality", fig))
        if "last_value" in metrics:
            reference = np.asarray(metrics["last_value"]["standardized"]["horizon_mse"], dtype=float)
            fig = _figure(f"Lead-wise error relative to persistence · full {split}", "MSE / persistence MSE")
            for name in ("model", "model_bounded"):
                if name in metrics:
                    values = np.asarray(metrics[name]["standardized"]["horizon_mse"], dtype=float)
                    ratios = np.divide(values, reference, out=np.full_like(values, np.nan), where=reference > 1e-12)
                    fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)), y=ratios, name=name))
            fig.add_hline(y=1.0, line_dash="dash")
            figures.append(("persistence_skill_by_lead", fig))
        audited = {name: result["range_audit"] for name, result in metrics.items()
                   if result.get("range_audit", {}).get("lower") is not None}
        if audited:
            fig = _figure(f"Physical range violations · all forecast cells · full {split}", "Percent")
            fig.add_trace(go.Bar(x=list(audited), y=[100 * r["below_fraction"] for r in audited.values()], name="Below lower bound"))
            fig.add_trace(go.Bar(x=list(audited), y=[100 * (r["above_fraction"] or 0) for r in audited.values()], name="Above upper bound"))
            figures.append(("range_violations", fig))
        values = np.array(metrics["model"]["standardized"]["channel_mse"], dtype=float)
        valid_indices = np.flatnonzero(np.isfinite(values))
        selected = valid_indices[np.argsort(values[valid_indices])[-min(30, len(valid_indices)):]]
        targets = state["data"]["targets"]
        fig = _figure(f"12 / Highest-error target channels · full {split}", "Target")
        fig.add_trace(go.Bar(x=values[selected], y=[targets[i] for i in selected], orientation="h"))
        fig.update_layout(height=max(360, 23 * len(selected)))
        fig.update_xaxes(title="Standardized MSE")
        figures.append(("channel_error", fig))
    examples = evaluation.get("examples", state.get("examples", []))
    for i, example in enumerate(examples):
        fig = _figure(f"Forecast / {example['channel']} · origin {example['origin']}", "Original units")
        lookback = len(example["history"])
        for name, xs, ys, times, dash in (
            ("History", list(range(-lookback, 0)), example["history"], example["history_time"], "solid"),
            ("Observed future", list(range(len(example["truth"]))), example["truth"], example["future_time"], "solid"),
            ("Forecast", list(range(len(example["prediction"]))), example["prediction"], example["future_time"], "dash"),
        ):
            fig.add_trace(go.Scatter(x=xs, y=ys, name=name, line=dict(dash=dash), customdata=times,
                                     hovertemplate="%{customdata}<br>%{y:.5g}<extra>%{fullData.name}</extra>"))
        if "prediction_bounded" in example:
            fig.add_trace(go.Scatter(x=list(range(len(example["prediction_bounded"]))),
                                     y=example["prediction_bounded"], name="Bounded forecast", line=dict(dash="dot")))
        fig.add_vline(x=-0.5, line_dash="dot", line_color="#64748b")
        fig.update_xaxes(title="Row offset from forecast origin")
        figures.append((f"forecast_{i:02d}", fig))
    return figures


def render_report(run_dir, export_png=False):
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "report_state.json").read_text(encoding="utf-8"))
    figures = make_figures(state)
    diag = state.get("diagnostic") or {}
    stats = diag.get("statistics", {})
    test_result = (state.get("evaluation") or {}).get("metrics", {}).get("model", {}).get("standardized", {})
    def format_number(number):
        return "Pending" if number is None else f"{number:.5g}"
    cfg = state["config"]
    rank_label = (f"{stats['effective_rank']:.3g} / {stats['rank_upper_bound']}"
                  if stats.get("available") else "Pending")
    cards = [("STATUS", state.get("status", "running")),
             ("BEST VAL MSE", format_number(state.get("best_val_mse"))),
             ("EVALUATION MSE", format_number(test_result.get("mse"))),
             ("RANK / UPPER BOUND", rank_label),
             ("MEAN CROSS-WINDOW STD", format_number(stats.get("mean_std"))),
             ("NEAR-ZERO DIMENSIONS", f"{100 * stats['near_zero_fraction']:.1f}%" if stats.get("available") else "Pending"),
             ("PROBE WINDOWS", str(stats.get("probe_count", "Pending"))),
             ("HISTORY → FUTURE", f"{cfg['lookback']} → {cfg['horizon']}"),
             ("PREDICTION MODE", cfg.get("prediction_mode", "direct")),
             ("FIRST STEP MSE / RAW", format_number(test_result.get("first_step_mse"))),
             ("NEAR MSE / RAW", format_number(test_result.get("near_mse"))),
             ("TRAINABLE PARAMETERS", str(state.get("parameter_count", "Not recorded")))]
    card_html = "".join(f'<div class="card"><span>{html.escape(k)}</span><strong>{html.escape(v)}</strong></div>'
                        for k, v in cards)
    sections = []
    for index, (name, figure) in enumerate(figures):
        block = figure.to_html(full_html=False, include_plotlyjs=True if index == 0 else False,
                               config={"responsive": True, "displaylogo": False,
                                       "toImageButtonOptions": {"format": "png", "scale": 2, "filename": name}})
        sections.append(f'<section class="plot" id="{name}">{block}</section>')
    if not sections:
        sections = ['<section class="note">尚无测量结果。训练开始后，本报告自动更新。</section>']
    notes = """<section class="note"><b>如何阅读</b><p>预测效果以按时间划分的验证集和测试集为准。
    训练曲线使用训练模式，验证使用评估模式，两者不必直接重合。有效秩来自跨样本、同时间位置中心化的协方差；
    “Temporal mean / Pooled” 只看窗口均值，可能丢失时序信息。近零阈值只是诊断参数，不是坍缩判决线。</p>
    <p>normal、zero、probe_mean、sample_shuffle、time_shuffle 使用相同探针窗口和冻结模型。
    probe_mean 是同位置的探针样本均值，仅用于诊断。干预可能造成分布偏移，误差变化不能单独证明因果关系。
    默认探针预测区间不重叠，历史窗口可能重叠；相似窗口并非严格独立。</p>
    <p>样本相似、低秩或预测平滑都不等于有害坍缩。标准化指标适合跨量纲汇总，原始量纲指标见 JSON/CSV。
    报告不使用模拟数据；未产生的结果保留为空。测试结果仅在最佳验证检查点选定后计算。</p></section>"""
    if cfg.get("prediction_mode") == "residual":
        notes += '''<section class="note"><b>残差模型的诊断</b><p>预测 = 原样本最后观测值 + 编码预测的修正。
        zero / probe_mean / sample_shuffle / time_shuffle 都保留原样本最后值；zero 仍包含预测头偏置。
        persistence_only 去掉整个学习修正；context_shuffle 只打乱最后值；joint_shuffle 同时打乱编码和最后值。
        编码干预影响小可能表示模型依赖持续性分支，不能直接判定坍缩。</p></section>'''
    if cfg.get("output_constraint", "none") != "none":
        notes += '''<section class="note"><b>原始输出与物理范围约束</b><p>model 始终是未经约束的原始预测；
        model_bounded 是同一模型在原始量纲上限制范围后的配对结果。训练和检查点选择使用原始预测，
        不用范围限制改善后的分数隐藏模型误差。越界比例覆盖完整评估的所有预测实例，不仅是图中示例。
        原始目标越界数另见 range_audit.csv，请核实容量上限和数据定义。</p></section>'''
    configuration = html.escape(json.dumps({"config": cfg, "data": state.get("data", {}),
                                            "provenance": state.get("provenance", {})},
                                           ensure_ascii=False, indent=2))
    error = f'<section class="note error">{html.escape(state["error"])}</section>' if state.get("error") else ""
    title = html.escape(Path(cfg["data"]).name)
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>Forecast Lab · {title}</title>
    <style>body{{margin:0;background:#f1f5f9;color:#17233b;font:15px/1.6 system-ui,sans-serif}}
    header{{background:#10213d;color:white;padding:36px max(24px,calc((100% - 1320px)/2))}}
    header small{{color:#7dd3fc;letter-spacing:3px}}h1{{font-size:32px;margin:8px 0}}header p{{color:#cbd5e1;margin:0}}
    main{{max-width:1320px;margin:auto;padding:24px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
    .card,.plot,.note,details{{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px}}
    .card span{{font-size:10px;letter-spacing:1px;color:#64748b;display:block}}.card strong{{font-size:23px;display:block;margin-top:8px}}
    .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:22px 0}}
    .plot{{padding:4px;overflow:hidden}}.note{{margin:18px 0}}.note p{{color:#526077}}.error{{color:#b91c1c}}
    pre{{overflow:auto;max-height:480px;font-size:12px}}footer{{color:#64748b;padding:20px 0}}
    @media(max-width:1000px){{.cards{{grid-template-columns:repeat(3,1fr)}}.grid{{grid-template-columns:1fr}}}}
    @media(max-width:520px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:12px}}h1{{font-size:24px}}}}
    </style></head><body><header><small>FORECAST LAB / REPRESENTATION OBSERVATORY</small>
    <h1>{title}</h1><p>直接预测 · 表征监测 · 编码干预 · 有证据的正则化</p></header>
    <main><div class="cards">{card_html}</div>{error}{notes}<div class="grid">{''.join(sections)}</div>
    <details><summary>实验配置、数据边界与运行环境</summary><pre>{configuration}</pre></details>
    <footer>离线报告 · 图表支持缩放、悬停与 PNG 下载 · 训练中刷新页面查看更新</footer></main></body></html>"""
    temporary = run_dir / "report.html.tmp"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(run_dir / "report.html")
    if export_png:
        render_pngs(state, run_dir / "figures")


def render_pngs(state, directory):
    """Matplotlib exports need no browser, display server or Kaleido install."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
    history = state.get("history", [])
    if history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r["epoch"] for r in history], [r["train_prediction"] for r in history], label="Train")
        ax.plot([r["epoch"] for r in history], [r["val_mse"] for r in history], label="Validation")
        ax.set(xlabel="Epoch", ylabel="Standardized MSE", title="Forecast learning curves")
        ax.legend()
        save(fig, "learning_curves")
    trajectory = state.get("monitor_history", [])
    if trajectory:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = [r["epoch"] for r in trajectory]
        axes[0].plot(x, [r.get("effective_rank") for r in trajectory], label="Matched-position")
        axes[0].plot(x, [r.get("pooled_effective_rank") for r in trajectory], label="Window mean")
        axes[0].set(title="Effective rank", xlabel="Epoch")
        axes[0].legend()
        axes[1].plot(x, [r.get("mean_std") for r in trajectory])
        axes[1].set(title="Cross-window std", xlabel="Epoch")
        save(fig, "representation_trajectory")
    diag = state.get("diagnostic") or {}
    stats = diag.get("statistics", {})
    if stats.get("available"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(np.arange(len(stats["std_per_dimension"])), stats["std_per_dimension"], color=COLORS[0])
        axes[0].axhline(stats["near_zero_threshold"], color=COLORS[4], linestyle="--")
        axes[0].set(title="Feature activity", xlabel="Feature", ylabel="Cross-window std")
        axes[1].semilogy(np.maximum(stats["eigenvalues"], 1e-12), label="Matched-position")
        axes[1].semilogy(np.maximum(stats["pooled_eigenvalues"], 1e-12), label="Window mean")
        axes[1].set(title="Covariance spectrum", xlabel="Eigenvalue index")
        axes[1].legend()
        save(fig, "activity_and_spectrum")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        im = axes[0].imshow(stats["correlation"], vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        axes[0].set(title="Feature correlation", xlabel="Feature", ylabel="Feature")
        fig.colorbar(im, ax=axes[0])
        im = axes[1].imshow(stats["pooled_cosine"], vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        axes[1].set(title="Window-mean cosine", xlabel="Probe window", ylabel="Probe window")
        fig.colorbar(im, ax=axes[1])
        save(fig, "correlations")
        fig, ax = plt.subplots(figsize=(6, 4))
        coords = np.asarray(stats["pooled_pca"])
        im = ax.scatter(coords[:, 0], coords[:, 1], c=diag["origins"], cmap="viridis")
        ax.set(title="Window-mean PCA", xlabel="PC 1", ylabel="PC 2")
        fig.colorbar(im, ax=ax, label="Origin row")
        save(fig, "pca")
    if diag.get("ablations"):
        fig, ax = plt.subplots(figsize=(9, 4))
        names = list(diag["ablations"])
        ax.bar(names, [diag["ablations"][k]["standardized"]["mse"] for k in names],
               color=[COLORS[i % len(COLORS)] for i in range(len(names))])
        ax.set(title=f"Latent interventions / {diag['split']} / epoch {diag['epoch']}", ylabel="Standardized MSE")
        ax.tick_params(axis="x", rotation=15)
        save(fig, "interventions")
    evaluation = state.get("evaluation") or {}
    if evaluation.get("metrics"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        metrics = evaluation["metrics"]
        names = list(metrics)
        axes[0].bar(names, [metrics[k]["standardized"]["mse"] for k in names], color=COLORS[:len(names)])
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].set(title=f"Full {evaluation['split']} split", ylabel="Standardized MSE")
        for name, result in metrics.items():
            values = result["standardized"]["horizon_mse"]
            axes[1].plot(np.arange(1, len(values) + 1), values, label=name)
        axes[1].set(title="Error by forecast lead", xlabel="Lead (rows)", ylabel="Standardized MSE")
        axes[1].legend()
        save(fig, "forecast_metrics")
        if "first_step_mse" in metrics["model"]["standardized"]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            positions = np.arange(3)
            width = 0.8 / len(metrics)
            for i, (name, result) in enumerate(metrics.items()):
                values = [result["standardized"].get(key) for key in ("first_step_mse", "near_mse", "far_mse")]
                axes[0].bar(positions + (i - (len(metrics) - 1) / 2) * width,
                            [np.nan if v is None else v for v in values], width, label=name)
                audit = result.get("range_audit", {})
                if audit.get("below_fraction") is not None:
                    axes[1].bar(name, 100 * (audit["below_fraction"] + (audit.get("above_fraction") or 0)))
            axes[0].set_xticks(positions, ["First step", "Near", "Far"])
            axes[0].set(ylabel="Standardized MSE", title="Near / far prediction")
            axes[0].legend(fontsize=8)
            axes[1].set(ylabel="Percent of all forecast cells", title="Outside configured physical range")
            axes[1].tick_params(axis="x", rotation=20)
            save(fig, "near_far_and_range")
    for i, example in enumerate(evaluation.get("examples", state.get("examples", []))):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(-len(example["history"]), 0), example["history"], label="History")
        ax.plot(example["truth"], label="Observed future")
        ax.plot(example["prediction"], "--", label="Forecast")
        if "prediction_bounded" in example:
            ax.plot(example["prediction_bounded"], ":", label="Bounded forecast")
        ax.axvline(-0.5, color="grey", linestyle=":")
        ax.set(title=f"{example['channel']} / origin {example['origin']}", xlabel="Row offset", ylabel="Original units")
        ax.legend()
        save(fig, f"forecast_{i:02d}")
