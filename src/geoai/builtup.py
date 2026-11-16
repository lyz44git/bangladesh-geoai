"""Chunked built-up mapping: labels, training samples and prediction rasters.

Large study areas do not fit in memory once ~30 feature layers are derived,
so everything here walks the composite in windows with a small halo (the
texture filters need neighbours beyond the window edge).
"""
from __future__ import annotations

import os

import numpy as np
import rasterio
from rasterio.windows import Window

from . import labels
from .composite import read_composite
from .features import feature_stack
from .grid import Grid

HALO = 8


def grid_of(path: str) -> Grid:
    with rasterio.open(path) as src:
        t = src.transform
        return Grid(t.c, t.f, src.width, src.height, t.a)


def windows(width: int, height: int, size: int = 1024, halo: int = HALO):
    """Yield (core_window, read_window, core_slice_within_read)."""
    for r in range(0, height, size):
        for c in range(0, width, size):
            core = Window(c, r, min(size, width - c), min(size, height - r))
            r0, c0 = max(0, r - halo), max(0, c - halo)
            r1, c1 = min(height, r + core.height + halo), min(width, c + core.width + halo)
            read = Window(c0, r0, c1 - c0, r1 - r0)
            sl = (slice(r - r0, r - r0 + core.height), slice(c - c0, c - c0 + core.width))
            yield core, read, sl


def window_features(path: str, read_win: Window, sl):
    refl, n_clear = read_composite(path, window=read_win)
    F, names = feature_stack(refl)
    return F[:, sl[0], sl[1]], names, n_clear[sl[0], sl[1]]


def write_label_raster(grid: Grid, out_path: str, classes=labels.BUILT, year: int = 2021, size: int = 2048):
    """WorldCover class fraction on `grid`, written as float32 GeoTIFF."""
    if os.path.exists(out_path):
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prof = dict(driver="GTiff", width=grid.width, height=grid.height, count=1, dtype="float32", crs=grid.crs,
                transform=grid.transform, nodata=np.nan, tiled=True, blockxsize=256, blockysize=256,
                compress="deflate")
    with rasterio.open(out_path + ".part", "w", **prof) as dst:
        for r, c, g in grid.chunks(size):
            dst.write(labels.class_fraction(g, classes, year=year), 1, window=Window(c, r, g.width, g.height))
    os.replace(out_path + ".part", out_path)
    return out_path


def read_window(path: str, win: Window) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1, window=win).astype(np.float32)


def predict_raster(path: str, out_path: str, predict_fn, size: int = 1024):
    """Apply predict_fn(F) -> prob[h, w] to every window; write uint8 (prob*250, 255 = no data)."""
    g = grid_of(path)
    prof = dict(driver="GTiff", width=g.width, height=g.height, count=1, dtype="uint8", crs=g.crs,
                transform=g.transform, nodata=255, tiled=True, blockxsize=256, blockysize=256, compress="deflate")
    with rasterio.open(out_path + ".part", "w", **prof) as dst:
        for core, read, sl in windows(g.width, g.height, size):
            F, _, _ = window_features(path, read, sl)
            p = predict_fn(F)
            dst.write(np.where(np.isfinite(p), np.round(p * 250), 255).astype(np.uint8), 1, window=core)
    os.replace(out_path + ".part", out_path)
    return out_path


def read_prob(path: str, window=None) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1, window=window)
    p = a.astype(np.float32) / 250.0
    p[a == 255] = np.nan
    return p
