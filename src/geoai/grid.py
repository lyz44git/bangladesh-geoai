"""Target grids and areas of interest.

All products share one projected CRS so that rasters from different
Sentinel-2 UTM zones (45N/46N) line up pixel-for-pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine, from_origin

# Transverse Mercator centred on 90E (WGS84). Bangladesh spans ~88.0E-92.7E,
# so scale distortion stays below ~0.1%, unlike a single UTM zone.
BD_TM = CRS.from_proj4(
    "+proj=tmerc +lat_0=0 +lon_0=90 +k=0.9996 +x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

_to_tm = Transformer.from_crs("EPSG:4326", BD_TM, always_xy=True)
_to_ll = Transformer.from_crs(BD_TM, "EPSG:4326", always_xy=True)


def lonlat_to_tm(lon, lat):
    return _to_tm.transform(lon, lat)


def tm_to_lonlat(x, y):
    return _to_ll.transform(x, y)


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid in BD_TM."""

    xmin: float
    ymax: float
    width: int
    height: int
    res: float

    @property
    def transform(self) -> Affine:
        return from_origin(self.xmin, self.ymax, self.res, self.res)

    @property
    def xmax(self) -> float:
        return self.xmin + self.width * self.res

    @property
    def ymin(self) -> float:
        return self.ymax - self.height * self.res

    @property
    def bounds(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def crs(self):
        return BD_TM

    def lonlat_bbox(self, pad_deg: float = 0.0):
        xs = np.array([self.xmin, self.xmax, self.xmin, self.xmax])
        ys = np.array([self.ymin, self.ymin, self.ymax, self.ymax])
        lon, lat = tm_to_lonlat(xs, ys)
        return (lon.min() - pad_deg, lat.min() - pad_deg, lon.max() + pad_deg, lat.max() + pad_deg)

    def chunks(self, size: int):
        """Yield (row_off, col_off, sub_grid) tiles of at most size x size pixels."""
        for r in range(0, self.height, size):
            for c in range(0, self.width, size):
                h = min(size, self.height - r)
                w = min(size, self.width - c)
                yield r, c, Grid(self.xmin + c * self.res, self.ymax - r * self.res, w, h, self.res)

    def coarsen(self, factor: int) -> "Grid":
        return Grid(self.xmin, self.ymax, self.width // factor, self.height // factor, self.res * factor)


def grid_from_lonlat(lon_min, lat_min, lon_max, lat_max, res: float, snap: float | None = None) -> Grid:
    """Grid covering a lon/lat box, origin snapped to a multiple of `snap` metres.

    Snapping to a common origin (default 1 km) makes grids built at different
    resolutions nest exactly (20 m pixels aggregate cleanly to 1 km cells).
    """
    snap = snap or 1000.0
    xs, ys = lonlat_to_tm([lon_min, lon_max, lon_min, lon_max], [lat_min, lat_min, lat_max, lat_max])
    xmin = np.floor(min(xs) / snap) * snap
    xmax = np.ceil(max(xs) / snap) * snap
    ymin = np.floor(min(ys) / snap) * snap
    ymax = np.ceil(max(ys) / snap) * snap
    return Grid(float(xmin), float(ymax), int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res)), res)


def grid_around(lon: float, lat: float, half_km: float, res: float) -> Grid:
    """Square grid of side 2*half_km centred on a point (snapped to 1 km)."""
    x, y = lonlat_to_tm(lon, lat)
    xmin = np.floor((x - half_km * 1000) / 1000) * 1000
    ymax = np.ceil((y + half_km * 1000) / 1000) * 1000
    n = int(round(2 * half_km * 1000 / res))
    return Grid(float(xmin), float(ymax), n, n, res)


def block_reduce_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Mean over non-overlapping factor x factor blocks, ignoring NaN."""
    h, w = a.shape[0] // factor * factor, a.shape[1] // factor * factor
    b = a[:h, :w].reshape(h // factor, factor, w // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))
