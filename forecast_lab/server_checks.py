"""Optional server-only integrity checks, NOT benchmark or quality claims.

Run explicitly on the target server: python server_checks.py
These checks were not executed on the author's local machine.
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tslab.config import Config
from tslab.data import load_data
from tslab.diagnostics import representation_statistics
from tslab.model import ForecastEncoder, masked_mse


class DataIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=120, freq="h"),
                                   "a": np.arange(120, dtype=float),
                                   "b": np.sin(np.arange(120, dtype=float))})
        self.path = self.root / "source.csv"
        self.frame.to_csv(self.path, index=False)
        self.cfg = Config(data=str(self.path), lookback=8, horizon=4, train_ratio=0.6, val_ratio=0.2)

    def tearDown(self):
        self.temporary.cleanup()

    def test_holdout_values_cannot_change_training_scaling_or_windows(self):
        first = load_data(self.cfg)
        changed = self.frame.copy()
        changed.loc[72:, ["a", "b"]] = 1e8
        changed_path = self.root / "changed.csv"
        changed.to_csv(changed_path, index=False)
        second = load_data(replace(self.cfg, data=str(changed_path)))
        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(first.scale, second.scale)
        np.testing.assert_array_equal(first.values[:72], second.values[:72])
        for split, dataset in first.datasets.items():
            lo, hi = first.bounds[split]
            self.assertTrue(np.all(dataset.origins >= lo))
            self.assertTrue(np.all(dataset.origins + self.cfg.horizon <= hi))

    def test_forward_fill_never_reads_the_next_observation(self):
        self.frame.loc[73, "a"] = np.nan
        self.frame.loc[74, "a"] = 100000.0
        self.frame.to_csv(self.path, index=False)
        bundle = load_data(replace(self.cfg, missing="ffill"))
        restored = bundle.values[73, 0] * bundle.scale[0] + bundle.mean[0]
        self.assertAlmostEqual(restored, 72.0, places=4)
        self.assertFalse(bundle.observed[73, 0])
        validation = bundle.datasets["val"]
        item = validation[int(np.flatnonzero(validation.origins == 72)[0])]
        self.assertFalse(bool(item["mask"][1, 0]))


class RepresentationIntegrity(unittest.TestCase):
    def test_position_only_latent_is_detected_as_cross_sample_constant(self):
        cfg = Config(d_model=8, probe_positions=8)
        temporal_pattern = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
        z = np.repeat(temporal_pattern, 16, axis=0)
        stats = representation_statistics(z, cfg)
        self.assertTrue(stats["available"])
        self.assertEqual(stats["mean_std"], 0.0)
        self.assertEqual(stats["effective_rank"], 0.0)
        self.assertEqual(stats["near_zero_fraction"], 1.0)

    def test_forecast_loss_reaches_encoder_and_ignores_unobserved_targets(self):
        torch.manual_seed(12)
        model = ForecastEncoder(3, 2, lookback=8, horizon=4, d_model=8, layers=2, dropout=0.0)
        x, y = torch.randn(6, 8, 3), torch.randn(6, 4, 2)
        mask = torch.ones_like(y, dtype=torch.bool)
        mask[:, 0] = False
        prediction, z, _ = model(x)
        loss = masked_mse(prediction, y, mask)
        changed = y.clone()
        changed[:, 0] = 1e5
        torch.testing.assert_close(loss, masked_mse(prediction, changed, mask))
        loss.backward()
        gradient = model.input_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertEqual(tuple(z.shape), (6, 8, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
