"""Forecast a new CSV's final window using a trained encoder; no future targets required."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tslab.engine import select_device
from tslab.config import Config
from tslab.constraints import constrain_original
from tslab.io import write_json
from tslab.model import ForecastEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-latent", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output must be new or empty")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    configuration = Config(**checkpoint["config"]).validate()
    cfg, metadata = vars(configuration), checkpoint["preprocessing"]
    features, targets = metadata["features"], metadata["targets"]
    frame = pd.read_csv(args.input, encoding=cfg["encoding"], low_memory=False)
    if not set(features).issubset(frame.columns):
        raise ValueError("Input must contain the same feature columns used in training")
    if len(frame) < cfg["lookback"]:
        raise ValueError(f"Need at least {cfg['lookback']} historical rows")
    time_column = metadata["time_column"]
    timestamps = None
    if time_column:
        if time_column not in frame:
            raise ValueError(f"Input is missing timestamp column {time_column}")
        timestamps = pd.to_datetime(frame[time_column], format="mixed", errors="raise")
        if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
            raise ValueError("Input timestamps must be valid, increasing and unique")
    raw = frame[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    if raw.isna().any().any() and cfg["missing"] == "error":
        raise ValueError("Missing data encountered; this checkpoint was trained with missing=error")
    mean, scale = np.array(metadata["mean"]), np.array(metadata["scale"])
    filled = raw.ffill().fillna(dict(zip(features, mean))).to_numpy(dtype=np.float64)
    normalized = ((filled[-cfg["lookback"]:] - mean) / scale).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite normalized history")
    device = select_device(args.device)
    model = ForecastEncoder(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        history_tensor = torch.from_numpy(normalized[None]).to(device)
        latent = model.encode(history_tensor)
        prediction = model.predict_from_latent(latent, model.context_from_input(history_tensor)).float().cpu().numpy()[0]
    target_indices = [features.index(name) for name in targets]
    prediction = prediction * scale[target_indices] + mean[target_indices]
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite forecast")
    raw_prediction = prediction.copy()
    prediction = constrain_original(prediction, configuration)
    result = pd.DataFrame(prediction, columns=targets)
    lead_column = "__forecast_lead__"
    while lead_column in result:
        lead_column = "_" + lead_column
    result.insert(0, lead_column, np.arange(1, cfg["horizon"] + 1))
    inferred_frequency = None
    if timestamps is not None and len(timestamps) >= 3:
        deltas = timestamps.diff().dropna()
        if deltas.nunique() == 1:
            delta = deltas.iloc[0]
            future = pd.date_range(timestamps.iloc[-1] + delta, periods=cfg["horizon"], freq=delta)
            result.insert(0, time_column, future.astype(str))
            inferred_frequency = str(delta)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "forecast.csv", index=False)
    if cfg["output_constraint"] != "none":
        raw_result = result.copy()
        raw_result[targets] = raw_prediction
        raw_result.to_csv(args.output / "forecast_raw.csv", index=False)
    if model.prediction_mode == "residual":
        # A latent alone is insufficient to reproduce a residual forecast.
        context_original = filled[-1, target_indices]
        write_json(args.output / "residual_context.json", {
            "targets": targets, "last_value_original": context_original.tolist(),
            "last_value_standardized": normalized[-1, target_indices].tolist()})
    if args.save_latent:
        np.save(args.output / "latent.npy", latent.float().cpu().numpy()[0])
    write_json(args.output / "prediction_metadata.json", {
        "checkpoint": str(args.checkpoint.resolve()), "input": str(args.input.resolve()),
        "checkpoint_epoch": checkpoint["epoch"], "lookback": cfg["lookback"], "horizon": cfg["horizon"],
        "features": features, "targets": targets, "latent_shape": list(latent.shape[1:]),
        "prediction_mode": model.prediction_mode, "output_constraint": cfg["output_constraint"],
        "output_min": cfg["output_min"], "output_max": cfg["output_max"],
        "frequency_from_entire_history": inferred_frequency,
        "note": "Training scaler reused unchanged. Missing/irregular clock spacing uses row leads only."})
    print(args.output / "forecast.csv")


if __name__ == "__main__":
    main()
