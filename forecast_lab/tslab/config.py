"""Configuration shared by training, evaluation and experiment suites."""
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    data: str = ""
    output: str = "runs/baseline"
    encoding: str = "utf-8"
    time_column: str = "auto"
    features: str = ""
    targets: str = ""
    missing: str = "error"
    split: str = "ratio"
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    lookback: int = 96
    horizon: int = 96
    train_stride: int = 1
    eval_stride: int = 1
    d_model: int = 64
    layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    patience: int = 7
    min_delta: float = 0.0
    seed: int = 42
    workers: int = 0
    device: str = "auto"
    amp: bool = False
    deterministic: bool = False
    tensorboard: bool = False
    monitor_every: int = 1
    probe_windows: int = 128
    probe_spacing: int = 0
    probe_positions: int = 16
    diag_batch_size: int = 16
    near_zero_threshold: float = 0.0001
    reconstruction_weight: float = 0.0
    variance_weight: float = 0.0
    covariance_weight: float = 0.0
    variance_target: float = 0.5
    seasonal_period: int = 0
    plot_channels: int = 3
    plot_windows: int = 3
    prediction_mode: str = "direct"
    near_steps: int = 12
    near_weight: float = 1.0
    output_constraint: str = "none"
    output_min: float = 0.0
    output_max: float = 16.0
    range_tolerance: float = 0.00001

    def validate(self):
        if not self.data:
            raise ValueError("Specify --data /path/to/dataset.csv")
        positive = ("lookback", "horizon", "train_stride", "eval_stride", "d_model",
                    "layers", "epochs", "batch_size", "patience", "monitor_every",
                    "probe_positions", "diag_batch_size", "plot_channels", "plot_windows", "near_steps")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.probe_windows < 2 or self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("probe_windows >= 2; kernel_size must be odd and >= 3")
        if self.d_model < 2:
            raise ValueError("d_model must be >= 2; LayerNorm of a single feature destroys input variation")
        if not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1
                and self.train_ratio + self.val_ratio < 1):
            raise ValueError("Split ratios must be positive and leave a nonempty test split")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("reconstruction_weight", "variance_weight", "covariance_weight",
                     "weight_decay", "min_delta", "workers", "probe_spacing", "seasonal_period"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.variance_target <= 0:
            raise ValueError("learning_rate, grad_clip and variance_target must be positive")
        if self.near_zero_threshold <= 0:
            raise ValueError("near_zero_threshold must be positive")
        if self.seasonal_period > self.lookback:
            raise ValueError("seasonal_period cannot exceed lookback")
        if self.split not in {"ratio", "ett-hour", "ett-minute"}:
            raise ValueError("split must be ratio, ett-hour or ett-minute")
        if self.missing not in {"error", "ffill"}:
            raise ValueError("missing must be error or ffill")
        if self.prediction_mode not in {"direct", "residual"}:
            raise ValueError("prediction_mode must be direct or residual")
        if not math.isfinite(self.near_weight) or self.near_weight <= 0:
            raise ValueError("near_weight must be positive; 1 means uniform loss")
        if self.output_constraint not in {"none", "nonnegative", "bounded"}:
            raise ValueError("output_constraint must be none, nonnegative or bounded")
        if not all(math.isfinite(v) for v in (self.output_min, self.output_max, self.range_tolerance)):
            raise ValueError("Output bounds and range_tolerance must be finite")
        if self.output_constraint == "bounded" and self.output_max <= self.output_min:
            raise ValueError("output_max must exceed output_min (in original units)")
        if self.range_tolerance < 0:
            raise ValueError("range_tolerance cannot be negative")
        if (self.variance_weight or self.covariance_weight) and self.batch_size < 2:
            raise ValueError("Representation regularization requires batch_size >= 2")
        return self


def parse_config():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args()
    defaults = asdict(Config())
    if known.config:
        overrides = json.loads(known.config.read_text(encoding="utf-8"))
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        defaults.update(overrides)
    parser = argparse.ArgumentParser(description="Train a forecast-first encoder", parents=[pre])
    for key, value in asdict(Config()).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=defaults[key])
        else:
            parser.add_argument(flag, type=type(value), default=defaults[key])
    values = vars(parser.parse_args())
    values.pop("config")
    return Config(**values).validate()
