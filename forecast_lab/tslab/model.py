"""Fixed temporal resolution encoder; every forecast passes through its latent."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, kernel, padding=dilation * (kernel - 1) // 2,
                              dilation=dilation)
        self.mix = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        h = self.conv(self.norm(z).transpose(1, 2)).transpose(1, 2)
        return z + self.dropout(self.mix(F.gelu(h)))


class ForecastEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, lookback, horizon,
                 d_model=64, layers=6, kernel_size=3, dropout=0.1, reconstruction=False):
        super().__init__()
        self.input_projection = nn.Linear(input_channels, d_model)
        position = torch.arange(lookback, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(lookback, d_model)
        pe[:, 0::2] = torch.sin(position * frequencies)
        pe[:, 1::2] = torch.cos(position * frequencies[:d_model // 2])
        self.register_buffer("position", pe.unsqueeze(0))
        self.blocks = nn.ModuleList([TemporalBlock(d_model, kernel_size, 2 ** (i % 6), dropout)
                                     for i in range(layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.time_projection = nn.Linear(lookback, horizon)
        self.output_projection = nn.Linear(d_model, output_channels)
        self.reconstruction_head = nn.Linear(d_model, input_channels) if reconstruction else None

    def encode(self, x):
        z = self.input_projection(x) + self.position
        for block in self.blocks:
            z = block(z)
        return self.final_norm(z)  # [batch, history time, latent feature]

    def predict_from_latent(self, z):
        h = self.time_projection(z.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(h)

    def forward(self, x):
        z = self.encode(x)
        prediction = self.predict_from_latent(z)
        reconstruction = self.reconstruction_head(z) if self.reconstruction_head is not None else None
        return prediction, z, reconstruction


def model_kwargs(cfg, bundle):
    return {"input_channels": len(bundle.features), "output_channels": len(bundle.targets),
            "lookback": cfg.lookback, "horizon": cfg.horizon, "d_model": cfg.d_model,
            "layers": cfg.layers, "kernel_size": cfg.kernel_size, "dropout": cfg.dropout,
            "reconstruction": cfg.reconstruction_weight > 0}


def masked_mse(prediction, target, mask):
    count = mask.sum()
    return ((prediction - target).square() * mask).sum() / count.clamp_min(1)


def representation_penalties(z, cfg):
    """Center across independent batch windows at MATCHED temporal positions.

    Do not treat positional embeddings or variation within one window as evidence
    of non-collapse. Both variance and covariance terms use cross-window variation.
    """
    zero = z.new_zeros((), dtype=torch.float32)
    if z.shape[0] < 2 or not (cfg.variance_weight or cfg.covariance_weight):
        return zero, zero
    indices = torch.linspace(0, z.shape[1] - 1, min(cfg.probe_positions, z.shape[1]),
                             device=z.device).long()
    h = z[:, indices].float()
    centered = h - h.mean(dim=0, keepdim=True)
    variance = centered.square().sum(dim=0) / (h.shape[0] - 1)
    var_loss = F.relu(cfg.variance_target - torch.sqrt(variance + 1e-4)).mean()
    if cfg.covariance_weight:
        covariance = torch.einsum("btd,bte->tde", centered, centered) / (h.shape[0] - 1)
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal(dim1=-2, dim2=-1))
        cov_loss = off_diagonal.square().sum() / (h.shape[1] * h.shape[2])
    else:
        cov_loss = zero
    return var_loss, cov_loss
