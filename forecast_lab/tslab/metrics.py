"""Streaming masked forecast metrics; no entire-test prediction buffer."""
import numpy as np
import torch


class MetricAccumulator:
    def __init__(self, horizon, target_scale):
        shape = (horizon, len(target_scale))
        self.count = np.zeros(shape, dtype=np.float64)
        self.squared = np.zeros(shape, dtype=np.float64)
        self.absolute = np.zeros(shape, dtype=np.float64)
        self.scale = np.asarray(target_scale, dtype=np.float64)

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

    def result(self):
        def divide(a, b):
            return np.divide(a, b, out=np.full_like(a, np.nan, dtype=np.float64), where=b > 0)
        count = self.count.sum()
        if count == 0:
            raise ValueError("No observed target cells available for evaluation")
        result = {"observed_forecast_cells": int(count)}
        for name, squared, absolute in (
            ("standardized", self.squared, self.absolute),
            ("original", self.squared * self.scale ** 2, self.absolute * self.scale),
        ):
            mse = float(squared.sum() / count)
            result[name] = {"mse": mse, "mae": float(absolute.sum() / count), "rmse": mse ** 0.5,
                            "horizon_mse": divide(squared.sum(axis=1), self.count.sum(axis=1)).tolist(),
                            "channel_mse": divide(squared.sum(axis=0), self.count.sum(axis=0)).tolist(),
                            "channel_mae": divide(absolute.sum(axis=0), self.count.sum(axis=0)).tolist()}
        return result


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
    target_scale = bundle.scale[bundle.target_indices]
    accumulators = {"model": MetricAccumulator(cfg.horizon, target_scale)}
    examples = []
    example_indices = set(np.linspace(0, len(loader.dataset) - 1,
                                      min(cfg.plot_windows, len(loader.dataset)), dtype=int).tolist())
    channels = np.unique(np.linspace(0, len(bundle.targets) - 1,
                                    min(cfg.plot_channels, len(bundle.targets)), dtype=int))
    offset = 0
    for batch in loader:
        x = batch["x"].to(device)
        prediction = model.predict_from_latent(model.encode(x))
        accumulators["model"].update(prediction, batch["y"], batch["mask"])
        if include_naive:
            for name, values in naive_predictions(x, bundle.target_indices, cfg.horizon,
                                                  cfg.seasonal_period).items():
                if name not in accumulators:
                    accumulators[name] = MetricAccumulator(cfg.horizon, target_scale)
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
        offset += len(x)
    return {k: v.result() for k, v in accumulators.items()}, examples
