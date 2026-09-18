"""Read-only probes. No gradients, weight updates or test-set model selection."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import MetricAccumulator


def _spectrum(covariance):
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
    total = eigenvalues.sum()
    if total <= 1e-20:
        return eigenvalues, 0.0
    probabilities = eigenvalues / total
    valid = probabilities > 0
    rank = float(np.exp(-(probabilities[valid] * np.log(probabilities[valid])).sum()))
    return eigenvalues, rank


def _correlation(covariance):
    denominator = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)


def representation_statistics(z, cfg):
    n, length, width = z.shape
    positions = np.unique(np.linspace(0, length - 1, min(cfg.probe_positions, length), dtype=int))
    if n < 2:
        return {"available": False, "reason": "At least two separated probe windows are required",
                "probe_count": n, "positions": positions.tolist()}
    sampled = z[:, positions].astype(np.float64)
    centered = sampled - sampled.mean(axis=0, keepdims=True)
    covariance = np.einsum("ntd,nte->de", centered, centered) / ((n - 1) * len(positions))
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    eigenvalues, rank = _spectrum(covariance)
    correlation = _correlation(covariance)
    # A second view retains only the temporal mean of each window. It is labeled
    # separately because mean pooling can discard otherwise informative dynamics.
    pooled = z.mean(axis=1).astype(np.float64)
    pooled_centered = pooled - pooled.mean(axis=0, keepdims=True)
    pooled_cov = pooled_centered.T @ pooled_centered / (n - 1)
    pooled_eigenvalues, pooled_rank = _spectrum(pooled_cov)
    _, _, vectors = np.linalg.svd(pooled_centered, full_matrices=False)
    coords = pooled_centered @ vectors[:2].T
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    norms = np.linalg.norm(pooled, axis=1)
    raw_unit = pooled / np.maximum(norms[:, None], 1e-12)
    pooled_cosine = raw_unit @ raw_unit.T
    offdiag = ~np.eye(n, dtype=bool)
    feature_offdiag = ~np.eye(width, dtype=bool)
    # Temporal cosine is an oversmoothing clue, not a collapse verdict.
    temporal_norm = np.linalg.norm(z, axis=-1, keepdims=True)
    unit = z / np.maximum(temporal_norm, 1e-12)
    adjacent = float((unit[:, 1:] * unit[:, :-1]).sum(axis=-1).mean()) if length > 1 else None
    return {"available": True, "probe_count": n, "positions": positions.tolist(),
            "mean_std": float(std.mean()), "std_per_dimension": std.tolist(),
            "near_zero_fraction": float((std < cfg.near_zero_threshold).mean()),
            "near_zero_threshold": cfg.near_zero_threshold,
            "effective_rank": rank, "rank_upper_bound": min(width, (n - 1) * len(positions)),
            "eigenvalues": eigenvalues.tolist(), "correlation": correlation.tolist(),
            "mean_absolute_correlation": float(np.abs(correlation[feature_offdiag]).mean()) if width > 1 else 0.0,
            "pooled_effective_rank": pooled_rank, "pooled_rank_upper_bound": min(width, n - 1),
            "pooled_eigenvalues": pooled_eigenvalues.tolist(),
            "pooled_cosine_mean": float(pooled_cosine[offdiag].mean()),
            "pooled_cosine": pooled_cosine.tolist(), "pooled_pca": coords.tolist(),
            "adjacent_token_cosine": adjacent}


@torch.inference_mode()
def run_diagnostics(model, dataset, bundle, cfg, device, epoch, split="val"):
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg.diag_batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(cfg.seed + 9107))
    chunks = []
    for batch in loader:
        chunks.append(model.encode(batch["x"].to(device)).float().cpu().numpy())
    z = np.concatenate(chunks, axis=0)
    statistics = representation_statistics(z, cfg)
    rng = np.random.default_rng(cfg.seed + 9107)
    # A shuffled cycle gives a reproducible derangement (no sample maps to itself).
    order = rng.permutation(len(z))
    donor = np.empty(len(z), dtype=int)
    donor[order] = np.roll(order, 1)
    temporal_order = rng.permutation(z.shape[1])
    reference = z.mean(axis=0, keepdims=True)
    modes = ["normal", "zero", "probe_mean", "sample_shuffle", "time_shuffle"]
    if len(z) < 2:
        modes.remove("sample_shuffle")
    target_scale = bundle.scale[bundle.target_indices]
    ablations = {}
    for mode in modes:
        accumulator = MetricAccumulator(cfg.horizon, target_scale)
        offset = 0
        for batch in loader:
            end = offset + len(batch["x"])
            if mode == "normal":
                hidden = z[offset:end]
            elif mode == "zero":
                hidden = np.zeros_like(z[offset:end])
            elif mode == "probe_mean":
                hidden = np.repeat(reference, end - offset, axis=0)
            elif mode == "sample_shuffle":
                hidden = z[donor[offset:end]]
            else:
                hidden = z[offset:end, temporal_order]
            prediction = model.predict_from_latent(torch.from_numpy(hidden.copy()).to(device))
            accumulator.update(prediction, batch["y"], batch["mask"])
            offset = end
        ablations[mode] = accumulator.result()
    reference_mse = ablations["normal"]["standardized"]["mse"]
    for result in ablations.values():
        mse = result["standardized"]["mse"]
        result["delta_mse"] = mse - reference_mse
        result["relative_delta"] = (mse - reference_mse) / reference_mse if reference_mse > 1e-12 else None
    return {"epoch": epoch, "split": split, "origins": dataset.origins.tolist(),
            "probe_spacing": cfg.probe_spacing or cfg.horizon,
            "statistics": statistics, "ablations": ablations,
            "notes": ["Fixed probes; evaluation mode; no gradients or parameter updates.",
                      "Targets do not overlap at default spacing; histories may overlap.",
                      "probe_mean is a diagnostic-only mean over probe inputs, not a deployable predictor.",
                      "Interventions are sensitivity checks and can shift the latent distribution.",
                      "Low rank, high cosine or low variance alone do not prove harmful collapse."]}
