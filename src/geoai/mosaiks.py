"""MOSAIKS-style random convolutional features (Rolf et al., 2021, Nature Communications).

One unsupervised featurisation of imagery that is reused for many prediction
tasks: K filters are random patches drawn from the imagery itself; each image
is summarised by the spatial mean of ReLU(+/- filter response - bias). A
linear (ridge) model on these features is then fitted per outcome.
"""
from __future__ import annotations

import numpy as np


class RandomConvFeatures:
    def __init__(self, n_filters: int = 512, ksize: int = 3, bias: float = 1.0, seed: int = 0):
        self.n_filters, self.ksize, self.bias, self.seed = n_filters, ksize, bias, seed
        self.filters = None
        self.mu = self.sd = None

    def fit(self, images: np.ndarray):
        """images: (N, C, H, W) float, NaN allowed. Draws filters from random image patches."""
        rng = np.random.default_rng(self.seed)
        flat = np.moveaxis(images, 1, 0).reshape(images.shape[1], -1)
        self.mu = np.nanmean(flat, 1).astype(np.float32)
        self.sd = (np.nanstd(flat, 1) + 1e-6).astype(np.float32)
        Z = self._standardise(images)
        n, c, h, w = Z.shape
        k = self.ksize
        filt = np.empty((self.n_filters, c, k, k), np.float32)
        for i in range(self.n_filters):
            j, r, s = rng.integers(n), rng.integers(h - k + 1), rng.integers(w - k + 1)
            p = Z[j, :, r:r + k, s:s + k]
            p = p - p.mean()  # contrast-normalise the patch
            filt[i] = p / (np.linalg.norm(p) + 1e-6)
        self.filters = filt
        return self

    def _standardise(self, images):
        Z = (images - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)

    def transform(self, images: np.ndarray, batch: int = 256) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(4)
        Wt = torch.from_numpy(self.filters)
        out = []
        with torch.no_grad():
            for s in range(0, len(images), batch):
                x = torch.from_numpy(self._standardise(images[s:s + batch]))
                r = F.conv2d(x, Wt)
                pos = F.relu(r - self.bias).mean((-2, -1))
                neg = F.relu(-r - self.bias).mean((-2, -1))
                out.append(torch.cat([pos, neg], 1).numpy())
        return np.concatenate(out)
