"""Per-pixel spectral and textural features from a 6-band composite.

Band order everywhere: blue, green, red, nir, swir16, swir22 (reflectance 0-1).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BAND_NAMES = ["blue", "green", "red", "nir", "swir16", "swir22"]


def _nd(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)


def indices(refl: np.ndarray) -> dict[str, np.ndarray]:
    blue, green, red, nir, sw1, sw2 = refl
    return {
        "ndvi": _nd(nir, red),
        "ndbi": _nd(sw1, nir),  # built-up / bare soil
        "mndwi": _nd(green, sw1),  # open water
        "bsi": _nd(sw1 + red, nir + blue),  # bare soil index
        "ndsw": _nd(sw1, sw2),
        "bright": np.nanmean(refl, axis=0),
    }


def _local_std(x: np.ndarray, size: int) -> np.ndarray:
    m = np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)
    mu = ndimage.uniform_filter(m, size)
    mu2 = ndimage.uniform_filter(m * m, size)
    return np.sqrt(np.clip(mu2 - mu * mu, 0, None))


def _local_mean(x: np.ndarray, size: int) -> np.ndarray:
    m = np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)
    return ndimage.uniform_filter(m, size)


def feature_stack(refl: np.ndarray, texture_sizes=(3, 7)) -> tuple[np.ndarray, list[str]]:
    """Return (F, h, w) float32 features and their names.

    Texture (local mean / std of indices) matters for built-up mapping in
    Bangladesh, where dense villages under tree cover are spectrally close to
    vegetation and only their heterogeneity separates them.
    """
    idx = indices(refl)
    feats = [refl[i] for i in range(refl.shape[0])]
    names = list(BAND_NAMES)
    for k, v in idx.items():
        feats.append(v)
        names.append(k)
    for k in ("ndvi", "ndbi", "mndwi", "bright"):
        for s in texture_sizes:
            feats.append(_local_std(idx[k], s))
            names.append(f"{k}_std{s}")
            feats.append(_local_mean(idx[k], s))
            names.append(f"{k}_mean{s}")
    F = np.stack(feats).astype(np.float32)
    F[:, ~np.isfinite(refl).all(0)] = np.nan
    return F, names
