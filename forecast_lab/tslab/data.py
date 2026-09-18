"""Chronological windows. Scaling fits training observations only."""
import warnings
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, values, observed, origins, lookback, horizon, target_indices):
        self.values = values
        self.observed = observed
        self.origins = np.asarray(origins, dtype=np.int64)
        self.lookback = lookback
        self.horizon = horizon
        self.target_indices = np.asarray(target_indices)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])
        x = self.values[origin - self.lookback:origin]
        y = self.values[origin:origin + self.horizon, self.target_indices]
        mask = self.observed[origin:origin + self.horizon, self.target_indices]
        x_mask = self.observed[origin - self.lookback:origin]
        return {"x": torch.from_numpy(x.copy()), "y": torch.from_numpy(y.copy()),
                "mask": torch.from_numpy(mask.copy()), "x_mask": torch.from_numpy(x_mask.copy()),
                "origin": origin}


@dataclass
class DataBundle:
    values: np.ndarray
    observed: np.ndarray
    timestamps: list
    features: list
    targets: list
    target_indices: list
    mean: np.ndarray
    scale: np.ndarray
    bounds: dict
    datasets: dict
    metadata: dict


def load_data(cfg, fitted=None):
    path = Path(cfg.data).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    source_sha256 = digest.hexdigest()
    if fitted is not None and fitted.get("source_sha256") and fitted["source_sha256"] != source_sha256:
        raise ValueError("Evaluation CSV content differs from the checkpoint's SHA-256 fingerprint")
    frame = pd.read_csv(path, encoding=cfg.encoding, low_memory=False)
    if frame.empty:
        raise ValueError("CSV is empty")
    if cfg.time_column == "auto":
        candidates = [c for c in frame.columns if c.lower() in {"date", "time", "timestamp", "datetime"}]
        time_column = candidates[0] if candidates else None
    else:
        time_column = None if cfg.time_column == "none" else cfg.time_column
    if time_column:
        if time_column not in frame:
            raise ValueError(f"Missing time column: {time_column}")
        dates = pd.to_datetime(frame[time_column], errors="raise", format="mixed")
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("Timestamps must be valid, unique and increasing; clean the CSV explicitly")
        if dates.diff().dropna().nunique() > 1:
            warnings.warn("Irregular timestamp intervals: horizons count ROWS, not fixed clock durations")
        timestamps = frame[time_column].astype(str).tolist()
    else:
        timestamps = [str(i) for i in range(len(frame))]
        warnings.warn("No timestamp column: preserving CSV row order; horizons count rows")
    features = cfg.features.split(",") if cfg.features else [c for c in frame.columns if c != time_column]
    features = [c.strip() for c in features]
    targets = [c.strip() for c in cfg.targets.split(",")] if cfg.targets else list(features)
    if not features or len(set(features)) != len(features) or len(set(targets)) != len(targets):
        raise ValueError("Feature/target selections must be nonempty and unique")
    if not set(features).issubset(frame.columns) or not set(targets).issubset(features):
        raise ValueError("All features must exist, and targets must be a subset of features")
    numeric = frame[features].apply(pd.to_numeric, errors="raise")
    raw = numeric.to_numpy(dtype=np.float64)
    raw[~np.isfinite(raw)] = np.nan
    n = len(raw)
    if cfg.split == "ratio":
        train_end = int(n * cfg.train_ratio)
        val_end = int(n * (cfg.train_ratio + cfg.val_ratio))
        test_end = n
    else:
        month = 30 * 24 * (4 if cfg.split == "ett-minute" else 1)
        train_end, val_end, test_end = 12 * month, 16 * month, 20 * month
        if n < test_end:
            raise ValueError(f"ETT split needs at least {test_end} rows; received {n}")
    raw = raw[:test_end]
    timestamps = timestamps[:test_end]
    timestamp_audit = {"available": False}
    if time_column:
        intervals = dates.iloc[:test_end].diff().dropna().dt.total_seconds()
        if len(intervals):
            typical = float(intervals.mode().iloc[0])
            timestamp_audit = {"available": True, "modal_interval_seconds": typical,
                               "min_interval_seconds": float(intervals.min()),
                               "max_interval_seconds": float(intervals.max()),
                               "different_interval_count": int((intervals != typical).sum()),
                               "used_first_time": timestamps[0], "used_last_time": timestamps[-1]}
    observed = np.isfinite(raw)
    if cfg.missing == "error" and not observed.all():
        raise ValueError("Missing/inf values found. Clean data or explicitly use --missing ffill")
    train_count = observed[:train_end].sum(axis=0)
    if (train_count < 2).any():
        raise ValueError("Every feature needs at least two observed TRAIN rows")
    if fitted is not None:
        for key, current in (("features", features), ("targets", targets), ("train_end", train_end),
                             ("val_end", val_end), ("test_end", test_end),
                             ("first_time", timestamps[0]), ("last_time", timestamps[-1])):
            if fitted[key] != current:
                raise ValueError(f"Evaluation CSV disagrees with training metadata: {key}")
        mean, scale = np.asarray(fitted["mean"]), np.asarray(fitted["scale"])
    else:
        mean = np.nanmean(raw[:train_end], axis=0)
        scale = np.nanstd(raw[:train_end], axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
    # Past observations may cross split boundaries; no backward fill or interpolation.
    filled = pd.DataFrame(raw).ffill().fillna(dict(enumerate(mean))).to_numpy()
    values = np.asarray((filled - mean) / scale, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scaled values: inspect feature ranges")
    indices = [features.index(c) for c in targets]
    bounds = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, test_end)}
    datasets = {}
    for name, (lo, hi) in bounds.items():
        stride = cfg.train_stride if name == "train" else cfg.eval_stride
        origins = np.arange(max(cfg.lookback, lo), hi - cfg.horizon + 1, stride, dtype=np.int64)
        # Exclude windows with no observed targets; masks still exclude individual missing cells.
        row_count = observed[:, indices].sum(axis=1)
        cumulative = np.concatenate(([0], np.cumsum(row_count)))
        origins = origins[cumulative[origins + cfg.horizon] - cumulative[origins] > 0]
        if not len(origins):
            raise ValueError(f"No usable {name} windows; reduce lookback/horizon or change split")
        datasets[name] = WindowDataset(values, observed, origins, cfg.lookback, cfg.horizon, indices)
    metadata = {"source": str(path), "source_rows": n, "used_rows": test_end,
                "source_sha256": source_sha256, "timestamp_audit": timestamp_audit,
                "features": features, "targets": targets, "time_column": time_column,
                "first_time": timestamps[0], "last_time": timestamps[-1],
                "train_end": train_end, "val_end": val_end, "test_end": test_end,
                "mean": mean.tolist(), "scale": scale.tolist(),
                "missing_cells": int((~observed).sum()), "missing_policy": cfg.missing,
                "windows": {k: len(v) for k, v in datasets.items()},
                "boundaries": {k: {"start_row": lo, "end_row_exclusive": hi,
                                      "first_time": timestamps[lo], "last_time": timestamps[hi - 1]}
                               for k, (lo, hi) in bounds.items()}}
    return DataBundle(values, observed, timestamps, features, targets, indices,
                      mean, scale, bounds, datasets, metadata)


def probe_dataset(bundle, cfg, split="val", sampling_seed=None):
    source = bundle.datasets[split]
    spacing = cfg.probe_spacing or cfg.horizon
    chosen = []
    for origin in source.origins:
        if not chosen or origin - chosen[-1] >= spacing:
            chosen.append(int(origin))
    if len(chosen) > cfg.probe_windows:
        if sampling_seed is None:
            positions = np.linspace(0, len(chosen) - 1, cfg.probe_windows, dtype=int)
        else:
            rng = np.random.default_rng(sampling_seed)
            positions = np.sort(rng.choice(len(chosen), cfg.probe_windows, replace=False))
        chosen = np.asarray(chosen)[positions].tolist()
    return WindowDataset(bundle.values, bundle.observed, chosen, cfg.lookback,
                         cfg.horizon, bundle.target_indices)
