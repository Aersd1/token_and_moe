"""Training and post-selection evaluation; test targets never select checkpoints."""
import logging
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_data, probe_dataset
from .diagnostics import run_diagnostics
from .io import write_csv, write_json
from .metrics import evaluate_forecasts
from .model import ForecastEncoder, masked_mse, model_kwargs, representation_penalties
from .report import render_report


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Install the appropriate CUDA PyTorch wheel")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda, cuda:N")
    return device


def _worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, cfg, device, train=False):
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=train, num_workers=cfg.workers,
                      pin_memory=device.type == "cuda", worker_init_fn=_worker_seed, generator=generator,
                      persistent_workers=cfg.workers > 0, drop_last=False)


def provenance(device):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=Path(__file__).resolve().parents[2],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"utc_started": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device), "git_commit": commit,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def save_checkpoint(path, model, cfg, bundle, epoch, score):
    checkpoint = {"format_version": 1, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "model_kwargs": model_kwargs(cfg, bundle), "config": asdict(cfg),
                  "preprocessing": bundle.metadata, "epoch": epoch, "val_mse": score}
    temporary = Path(path).with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def monitor_row(diagnostic):
    row = {"epoch": diagnostic["epoch"], "split": diagnostic["split"]}
    for key, value in diagnostic["statistics"].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[key] = value
    for mode, result in diagnostic["ablations"].items():
        row[f"{mode}_mse"] = result["standardized"]["mse"]
        row[f"{mode}_delta_mse"] = result["delta_mse"]
    return row


def export_evaluation(directory, metrics, examples, bundle):
    directory = Path(directory)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "forecast_examples.json", examples)
    rows = []
    channels = []
    horizon_rows = []
    for model, result in metrics.items():
        for space in ("standardized", "original"):
            values = result[space]
            rows.append({"model": model, "space": space,
                         **{k: values[k] for k in ("mse", "mae", "rmse")},
                         "observed_forecast_cells": result["observed_forecast_cells"]})
            for index, channel in enumerate(bundle.targets):
                channels.append({"model": model, "space": space, "channel": channel,
                                 "mse": values["channel_mse"][index], "mae": values["channel_mae"][index]})
            for index, mse in enumerate(values["horizon_mse"]):
                horizon_rows.append({"model": model, "space": space, "lead": index + 1, "mse": mse})
    write_csv(directory / "metrics.csv", rows)
    write_csv(directory / "channel_metrics.csv", channels)
    write_csv(directory / "horizon_metrics.csv", horizon_rows)
    prediction_rows = []
    for example in examples:
        for lead, (timestamp, actual, predicted) in enumerate(zip(example["future_time"], example["truth"], example["prediction"]), 1):
            prediction_rows.append({"origin": example["origin"], "channel": example["channel"],
                                    "lead": lead, "timestamp": timestamp,
                                    "observed": actual, "forecast": predicted})
    write_csv(directory / "forecast_examples.csv", prediction_rows)


def train(cfg):
    cfg.validate()
    output = Path(cfg.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    cfg.output = str(output)
    cfg.data = str(Path(cfg.data).expanduser().resolve())
    logger = logging.getLogger("forecast_lab")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "train.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    state = {"status": "initializing", "config": asdict(cfg), "history": [], "monitor_history": [],
             "diagnostic": None, "evaluation": None, "best_val_mse": None}
    writer = None
    write_json(output / "config.json", asdict(cfg))
    try:
        seed_everything(cfg.seed, cfg.deterministic)
        device = select_device(cfg.device)
        state["provenance"] = provenance(device)
        bundle = load_data(cfg)
        state["data"] = bundle.metadata
        write_json(output / "data_metadata.json", bundle.metadata)
        write_json(output / "provenance.json", state["provenance"])
        loaders = {name: make_loader(ds, cfg, device, train=name == "train")
                   for name, ds in bundle.datasets.items()}
        probe = probe_dataset(bundle, cfg, "val")
        write_json(output / "probe_origins.json", {"split": "val", "origins": probe.origins.tolist(),
                                                    "spacing": cfg.probe_spacing or cfg.horizon})
        model = ForecastEncoder(**model_kwargs(cfg, bundle)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(1, cfg.patience // 3))
        use_amp = cfg.amp and device.type == "cuda"
        # This interface also supports the minimum PyTorch 2.2 version.
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        if cfg.amp and not use_amp:
            logger.warning("AMP requested on CPU; using float32")
        if cfg.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(output / "tensorboard"))
        logger.info("Device=%s parameters=%d train/val/test windows=%s", device,
                    sum(p.numel() for p in model.parameters()), bundle.metadata["windows"])
        logger.info("Validation probes=%d; raw data are not copied into the run", len(probe))
        if cfg.variance_weight or cfg.covariance_weight:
            if len(bundle.datasets["train"]) < 2:
                raise ValueError("Regularization needs at least two training windows")
            logger.info("Singleton final batches skip variance/covariance regularizers")
        # Epoch zero is an actual measurement of the initialized model.
        state["status"] = "running"
        initial = run_diagnostics(model, probe, bundle, cfg, device, epoch=0)
        state["diagnostic"] = initial
        state["monitor_history"].append(monitor_row(initial))
        write_json(output / "diagnostics" / "epoch_0000.json", initial)
        write_json(output / "report_state.json", state)
        render_report(output)
        best = float("inf")
        patience_best = float("inf")
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, cfg.epochs + 1):
            start = time.perf_counter()
            model.train()
            squared_error_sum = 0.0
            target_count = 0
            total_weight = 0
            sums = {"total": 0.0, "reconstruction": 0.0, "variance": 0.0, "covariance": 0.0}
            for batch in loaders["train"]:
                x, y, mask = (batch[k].to(device, non_blocking=True) for k in ("x", "y", "mask"))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    prediction, z, reconstruction = model(x)
                    prediction_loss = masked_mse(prediction.float(), y, mask)
                    rec_loss = (masked_mse(reconstruction.float(), x, batch["x_mask"].to(device))
                                if reconstruction is not None else prediction_loss.new_zeros(()))
                    # Covariance must stay float32 even under mixed precision.
                    with torch.autocast(device_type=device.type, enabled=False):
                        var_loss, cov_loss = representation_penalties(z.float(), cfg)
                    loss = (prediction_loss + cfg.reconstruction_weight * rec_loss
                            + cfg.variance_weight * var_loss + cfg.covariance_weight * cov_loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}; try smaller LR / no AMP")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # GradScaler handles overflow by skipping the step under AMP.
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=not use_amp)
                scaler.step(optimizer)
                scaler.update()
                observed = int(mask.sum().item())
                squared_error_sum += float(prediction_loss.detach()) * observed
                target_count += observed
                batch_size = len(x)
                total_weight += batch_size
                for name, value in (("total", loss), ("reconstruction", rec_loss),
                                     ("variance", var_loss), ("covariance", cov_loss)):
                    sums[name] += float(value.detach()) * batch_size
            validation, examples = evaluate_forecasts(model, loaders["val"], bundle, cfg, device, include_naive=False)
            score = validation["model"]["standardized"]["mse"]
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation score")
            scheduler.step(score)
            if score < best:
                best, best_epoch = score, epoch
                save_checkpoint(output / "best.pt", model, cfg, bundle, epoch, score)
            if score < patience_best - cfg.min_delta:
                patience_best, stale_epochs = score, 0
            else:
                stale_epochs += 1
            row = {"epoch": epoch, "train_prediction": squared_error_sum / target_count,
                   **{f"train_{k}": v / total_weight for k, v in sums.items()},
                   "val_mse": score, "val_mae": validation["model"]["standardized"]["mae"],
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "epoch_seconds_before_monitor": time.perf_counter() - start}
            state["history"].append(row)
            state["examples"] = examples
            state["best_val_mse"] = best
            state["best_epoch"] = best_epoch
            stopping = stale_epochs >= cfg.patience
            if epoch % cfg.monitor_every == 0 or epoch == cfg.epochs or stopping:
                diagnostic = run_diagnostics(model, probe, bundle, cfg, device, epoch)
                state["diagnostic"] = diagnostic
                state["monitor_history"].append(monitor_row(diagnostic))
                write_json(output / "diagnostics" / f"epoch_{epoch:04d}.json", diagnostic)
            write_csv(output / "history.csv", state["history"])
            write_csv(output / "monitor_history.csv", state["monitor_history"])
            write_json(output / "report_state.json", state)
            render_report(output)
            logger.info("Epoch %d train=%.6f val=%.6f best=%.6f (epoch %d)", epoch,
                        row["train_prediction"], score, best, best_epoch)
            if writer:
                for key, value in row.items():
                    if key != "epoch":
                        writer.add_scalar(f"training/{key}", value, epoch)
                if state["diagnostic"]["epoch"] == epoch:
                    for key, value in state["monitor_history"][-1].items():
                        if key not in {"epoch", "split"} and isinstance(value, (int, float)):
                            writer.add_scalar(f"monitor/{key}", value, epoch)
                writer.flush()
            if stopping:
                logger.info("Early stopping based only on validation MSE")
                break
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        # The final dashboard shows the selected checkpoint, never the last epoch by accident.
        selected_diagnostic = run_diagnostics(model, probe, bundle, cfg, device, best_epoch)
        write_json(output / "diagnostics" / "best_validation.json", selected_diagnostic)
        state["diagnostic"] = selected_diagnostic
        val_metrics, val_examples = evaluate_forecasts(model, loaders["val"], bundle, cfg, device)
        export_evaluation(output / "validation", val_metrics, val_examples, bundle)
        logger.info("Checkpoint selected. Evaluating test split once.")
        test_metrics, test_examples = evaluate_forecasts(model, loaders["test"], bundle, cfg, device)
        export_evaluation(output / "test", test_metrics, test_examples, bundle)
        state["evaluation"] = {"split": "test", "epoch": best_epoch, "metrics": test_metrics, "examples": test_examples}
        state["status"] = "complete"
        summary = {"status": "complete", "config": asdict(cfg), "best_epoch": best_epoch,
                   "best_val_mse": best, "validation": val_metrics, "test": test_metrics,
                   "diagnostic": monitor_row(selected_diagnostic), "provenance": state["provenance"]}
        write_json(output / "report_state.json", state)
        render_report(output, export_png=True)
        # A suite may skip this run only after evaluation AND artifact export succeeded.
        write_json(output / "summary.json", summary)
        logger.info("Test MSE=%.6f; offline report: %s", test_metrics["model"]["standardized"]["mse"], output / "report.html")
        return summary
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "report_state.json", state)
        write_json(output / "failure.json", {"status": state["status"], "error": state["error"]})
        logger.exception("Run did not complete")
        try:
            render_report(output)
        except Exception:
            logger.exception("Could not render failure report; inspect failure.json")
        raise
    finally:
        if writer:
            writer.close()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
