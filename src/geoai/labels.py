"""Reference labels: ESA WorldCover 10 m land cover (AWS open data).

WorldCover classes: 10 tree, 20 shrub, 30 grass, 40 cropland, 50 built-up,
60 bare/sparse, 70 snow/ice, 80 permanent water, 90 herbaceous wetland,
95 mangroves, 100 moss/lichen.
"""
from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from .composite import GDAL_ENV
from .grid import Grid

WC_URL = {
    2020: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif",
    2021: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif",
}
BUILT = 50
WATER = 80


def wc_tiles(bbox_lonlat):
    """WorldCover 3x3 degree tile names (named by their south-west corner)."""
    lon0, lat0, lon1, lat1 = bbox_lonlat
    out = []
    for la in range(int(math.floor(lat0 / 3) * 3), int(math.floor(lat1 / 3) * 3) + 1, 3):
        for lo in range(int(math.floor(lon0 / 3) * 3), int(math.floor(lon1 / 3) * 3) + 1, 3):
            out.append(f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}")
    return out


def class_fraction(grid: Grid, classes, year: int = 2021) -> np.ndarray:
    """Fraction of each target pixel covered by `classes` (NaN where no data)."""
    classes = np.atleast_1d(classes)
    num = np.zeros((grid.height, grid.width), np.float32)
    den = np.zeros((grid.height, grid.width), np.float32)
    for tile in wc_tiles(grid.lonlat_bbox(pad_deg=0.01)):
        href = WC_URL[year].format(tile=tile)
        with rasterio.Env(**GDAL_ENV):
            try:
                src = rasterio.open(href)
            except rasterio.errors.RasterioIOError:
                continue  # ocean tiles do not exist
            with src:
                l, b, r, t = transform_bounds(grid.crs, src.crs, *grid.bounds, densify_pts=21)
                try:
                    win = from_bounds(l, b, r, t, src.transform).intersection(Window(0, 0, src.width, src.height))
                    win = Window(int(win.col_off), int(win.row_off), int(math.ceil(win.width)) + 1,
                                 int(math.ceil(win.height)) + 1).intersection(Window(0, 0, src.width, src.height))
                except rasterio.errors.WindowError:
                    continue
                if win.width < 1 or win.height < 1:
                    continue
                # Read at <= ~half the target resolution so the class fraction is meaningful.
                deg_target = grid.res / 111_000.0
                factor = max(1, int(deg_target / abs(src.res[0]) / 2))
                oh, ow = max(1, int(win.height // factor)), max(1, int(win.width // factor))
                lab = src.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.nearest)
                tr = src.window_transform(win) * rasterio.Affine.scale(win.width / ow, win.height / oh)
                crs = src.crs
        valid = (lab > 0).astype(np.float32)
        hit = np.isin(lab, classes).astype(np.float32)
        for arr, acc in ((hit, num), (valid, den)):
            dst = np.zeros_like(acc)
            reproject(arr, dst, src_transform=tr, src_crs=crs, dst_transform=grid.transform, dst_crs=grid.crs,
                      resampling=Resampling.average, src_nodata=None, dst_nodata=0)
            acc += dst
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = num / den
    frac[den < 0.5] = np.nan
    return frac
