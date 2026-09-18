"""Streaming masked forecast metrics; no entire-test prediction buffer."""
import numpy as np
import torch

from .constraints import constrain_standardized, physical_bounds


class MetricAccumulator:
    def __init__(self, horizon, target_scale, target_mean=None, near_steps=12,
                 lower=None, upper=None, tolerance=1e-5):
        shape = (horizon, len(target_scale))
        self.count = np.zeros(shape, dtype=np.float64)
        self.squared = np.zeros(shape, dtype=np.float64)
        self.absolute = np.zeros(shape, dtype=np.float64)
        self.scale = np.asarray(target_scale, dtype=np.float64)
        self.mean = np.zeros_like(self.scale) if target_mean is None else np.asarray(target_mean, dtype=np.float64)
        self.near_steps = min(near_steps, horizon)
        self.lower, self.upper, self.tolerance = lower, upper, tolerance
        self.predicted_count = self.below = self.above = self.target_below = self.target_above = 0
        self.minimum, self.maximum = float("inf"), -float("inf")

    def update(self, prediction, target, mask):
        def array(x):
            return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        p, t, m = array(prediction), array(target), array(mask).astype(bool)
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite forecast encountered")
        error = p.astype(np.float64) - t.astype(np.float64)
        error = np.where(m, error, 0.0)
        self.count += m.sum(axis=0)
        self.squared += (error ** 2).sum(axis=0)
        self.absolute += np.abs(error).sum(axis=0)
        original = p.astype(np.float64) * self.scale + self.mean
        actual = t.astype(np.float64) * self.scale + self.mean
        self.predicted_count += original.size
        self.minimum = min(self.minimum, float(original.min()))
        self.maximum = max(self.maximum, float(original.max()))
        if self.lower is not None:
            self.below += int((original < self.lower - self.tolerance).sum())
            self.target_below += int(((actual < self.lower - self.tolerance) & m).sum())
        if self.upper is not None:
            self.above += int((original > self.upper + self.tolerance).sum())
            self.target_above += int(((actual > self.upper + self.tolerance) & m).sum())

    def result(self):
        def divide(a, b):
            return np.divide(a, b, out=np.full_like(a, np.nan, dtype=np.float64), where=b > 0)
        count = self.count.sum()
        if count == 0:
            raise ValueError("No observed target cells available for evaluation")
        result = {"observed_forecast_cells": int(count), "near_steps": self.near_steps,
                  "range_audit": {"lower": self.lower, "upper": self.upper,
                                  "tolerance": self.tolerance, "predicted_cells": self.predicted_count,
                                  "min_prediction": self.minimum, "max_prediction": self.maximum,
                                  "below_count": self.below if self.lower is not None else None,
                                  "above_count": self.above if self.upper is not None else None,
                                  "below_fraction": self.below / self.predicted_count if self.lower is not None else None,
                                  "above_fraction": self.above / self.predicted_count if self.upper is not None else None,
                                  "observed_targets_below": self.target_below if self.lower is not None else None,
                                  "observed_targets_above": self.target_above if self.upper is not None else None}}
        for name, squared, absolute in (
            ("standardized", self.squared, self.absolute),
            ("original", self.squared * self.scale ** 2, self.absolute * self.scale),
        ):
            mse = float(squared.sum() / count)
            result[name] = {"mse": mse, "mae": float(absolute.sum() / count), "rmse": mse ** 0.5,
                            "horizon_mse": divide(squared.sum(axis=1), self.count.sum(axis=1)).tolist(),
                            "channel_mse": divide(squared.sum(axis=0), self.count.sum(axis=0)).tolist(),
                            "channel_mae": divide(absolute.sum(axis=0), self.count.sum(axis=0)).tolist()}
            for key, region in (("first_step", slice(0, 1)), ("near", slice(0, self.near_steps)),
                                ("far", slice(self.near_steps, None))):
                cells = self.count[region].sum()
                result[name][f"{key}_mse"] = float(squared[region].sum() / cells) if cells else None
                result[name][f"{key}_mae"] = float(absolute[region].sum() / cells) if cells else None
        return result


def make_accumulator(bundle, cfg):
    lower, upper = physical_bounds(cfg)
    indices = bundle.target_indices
    return MetricAccumulator(cfg.horizon, bundle.scale[indices], bundle.mean[indices],
                             cfg.near_steps, lower, upper, cfg.range_tolerance)


def naive_predictions(x, indices, horizon, seasonal_period=0):
    history = x[:, :, indices]
    result = {"last_value": history[:, -1:, :].expand(-1, horizon, -1),
              "history_mean": history.mean(dim=1, keepdim=True).expand(-1, horizon, -1)}
    if seasonal_period:
        positions = torch.arange(horizon, device=x.device) % seasonal_period
        result["seasonal_naive"] = history[:, -seasonal_period:, :][:, positions, :]
    return result


@torch.inference_mode()
def evaluate_forecasts(model, loader, bundle, cfg, device, include_naive=True):
    model.eval()
    accumulators = {"model": make_accumulator(bundle, cfg)}
    if cfg.output_constraint != "none":
        accumulators["model_bounded"] = make_accumulator(bundle, cfg)
    examples = []
    example_indices = set(np.linspace(0, len(loader.dataset) - 1,
                                      min(cfg.plot_windows, len(loader.dataset)), dtype=int).tolist())
    channels = np.unique(np.linspace(0, len(bundle.targets) - 1,
                                    min(cfg.plot_channels, len(bundle.targets)), dtype=int))
    offset = 0
    for batch in loader:
        x = batch["x"].to(device)
        prediction = model.predict_from_latent(model.encode(x), model.context_from_input(x))
        accumulators["model"].update(prediction, batch["y"], batch["mask"])
        bounded = None
        if "model_bounded" in accumulators:
            bounded = constrain_standardized(prediction, bundle, cfg)
            accumulators["model_bounded"].update(bounded, batch["y"], batch["mask"])
        if include_naive:
            for name, values in naive_predictions(x, bundle.target_indices, cfg.horizon,
                                                  cfg.seasonal_period).items():
                if name not in accumulators:
                    accumulators[name] = make_accumulator(bundle, cfg)
                accumulators[name].update(values, batch["y"], batch["mask"])
        for local in range(len(x)):
            if offset + local not in example_indices:
                continue
            origin = int(batch["origin"][local])
            for channel in channels:
                feature_index = bundle.target_indices[channel]
                scale, mean = bundle.scale[feature_index], bundle.mean[feature_index]
                truth = batch["y"][local, :, channel].numpy() * scale + mean
                truth = np.where(batch["mask"][local, :, channel].numpy(), truth, np.nan)
                examples.append({"origin": origin, "channel": bundle.targets[channel],
                                 "history": (batch["x"][local, :, feature_index].numpy() * scale + mean).tolist(),
                                 "truth": truth.tolist(),
                                 "prediction": (prediction[local, :, channel].float().cpu().numpy() * scale + mean).tolist(),
                                 "history_time": bundle.timestamps[origin - cfg.lookback:origin],
                                 "future_time": bundle.timestamps[origin:origin + cfg.horizon]})
                if bounded is not None:
                    examples[-1]["prediction_bounded"] = (bounded[local, :, channel].float().cpu().numpy() * scale + mean).tolist()
        offset += len(x)
    return {k: v.result() for k, v in accumulators.items()}, examples
