"""Physical limits are explicit postprocessing, not a hidden change to training.

Raw forecasts ALWAYS select checkpoints. Bounded forecasts are a paired additional
evaluation/deployment output. Bounds are specified in ORIGINAL target units.
"""
import numpy as np
import torch


def physical_bounds(cfg):
    if cfg.output_constraint == "nonnegative":
        return 0.0, None
    if cfg.output_constraint == "bounded":
        return cfg.output_min, cfg.output_max
    return None, None


def constrain_original(values, cfg):
    lower, upper = physical_bounds(cfg)
    if lower is None and upper is None:
        return values.copy()
    return np.clip(values, -np.inf if lower is None else lower, np.inf if upper is None else upper)


def constrain_standardized(prediction, bundle, cfg):
    lower, upper = physical_bounds(cfg)
    if lower is None and upper is None:
        return prediction
    mean = prediction.new_tensor(bundle.mean[bundle.target_indices])
    scale = prediction.new_tensor(bundle.scale[bundle.target_indices])
    result = prediction
    if lower is not None:
        result = torch.maximum(result, (lower - mean) / scale)
    if upper is not None:
        result = torch.minimum(result, (upper - mean) / scale)
    return result
