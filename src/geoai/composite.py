"""Cloud-masked median composites of Sentinel-2 L2A on an arbitrary target grid.

Scenes are read straight from the public Cloud-Optimized GeoTIFFs on AWS.
Only the window that overlaps the target chunk is fetched, at the overview
level closest to the target resolution, so a 20 m composite never downloads
full 10 m tiles.
"""
from __future__ import annotations

import logging
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from . import stac
from .grid import Grid

log = logging.getLogger(__name__)

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="6",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
    GDAL_CACHEMAX=512,
)
os.environ.setdefault("CURL_CA_BUNDLE", os.environ.get("SSL_CERT_FILE", ""))

# Scene Classification Layer classes kept as clear observations:
# 4 vegetation, 5 not-vegetated, 6 water, 7 unclassified, 11 snow.
# Dropped: 0 no data, 1 saturated, 2 dark/topographic shadow, 3 cloud shadow,
# 8/9 cloud (medium/high probability), 10 thin cirrus.
SCL_CLEAR = np.array([4, 5, 6, 7, 11])


def read_to_grid(href: str, grid: Grid, categorical: bool = False) -> np.ndarray | None:
    """Read one remote band onto `grid`. Returns float32 with NaN for no data.

    Returns None when the window misses the scene or the asset cannot be read
    (some early L2A items point at requester-pays JPEG2000 instead of public COGs).
    """
    if not href.startswith("http"):
        return None
    try:
        return _read_to_grid(href, grid, categorical)
    except rasterio.errors.WindowError:
        return None  # scene footprint does not reach this chunk
    except rasterio.errors.RasterioIOError as e:
        log.warning("skip unreadable asset %s (%s)", href, str(e)[:80])
        return None


def _read_to_grid(href: str, grid: Grid, categorical: bool) -> np.ndarray | None:
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            l, b, r, t = transform_bounds(grid.crs, src.crs, *grid.bounds, densify_pts=21)
            pad = 2 * src.res[0]
            win = from_bounds(l - pad, b - pad, r + pad, t + pad, src.transform)
            win = win.intersection(Window(0, 0, src.width, src.height))
            win = Window(int(win.col_off), int(win.row_off), int(np.ceil(win.width)), int(np.ceil(win.height)))
            if win.width < 1 or win.height < 1:
                return None
            factor = max(1.0, grid.res / src.res[0])
            out_h = max(1, int(round(win.height / factor)))
            out_w = max(1, int(round(win.width / factor)))
            data = src.read(
                1,
                window=win,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest if categorical else Resampling.average,
            ).astype(np.float32)
            src_tr = src.window_transform(win) * rasterio.Affine.scale(win.width / out_w, win.height / out_h)
            src_crs = src.crs
    if not np.any(data):
        return None
    dst = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        _reproject(data, dst, src_tr, src_crs, grid, categorical)
    return dst


def _reproject(data, dst, src_tr, src_crs, grid, categorical):
    reproject(
        data,
        dst,
        src_transform=src_tr,
        src_crs=src_crs,
        src_nodata=0,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest if categorical else Resampling.bilinear,
    )


def select_scenes(items, per_tile: int, year: int | None = None, by_orbit: bool = False):
    """Pick scenes per MGRS tile so that every year samples the season the same way.

    The dry season is split into `per_tile` equal date windows and the least
    cloudy scene of each window is kept ("phenology-balanced" selection). In
    Bangladesh the land surface changes fast within the season (bare fields
    after the aman harvest in Nov-Dec, flooded boro rice in Jan-Feb), so
    simply taking the N least cloudy scenes gives each year a different date
    mix, and the classifier then sees year-specific spectra that look like
    built-up change. Windows without a usable scene are back-filled with the
    least cloudy remaining scenes of that tile. With year=None the N least
    cloudy scenes are taken (the original behaviour). With by_orbit=True the
    selection is made separately for each (tile, relative orbit), so a tile
    imaged from two partially overlapping orbits gets scenes from both.
    """
    by_tile = defaultdict(list)
    for it in items:  # items arrive sorted by cloud cover
        by_tile[(stac.tile_id(it), stac.rel_orbit(it) if by_orbit else "")].append(it)
    if year is None:
        return [it for v in by_tile.values() for it in v[:per_tile]]
    start, end = (np.datetime64(d) for d in stac.dry_season(year))
    span = (end - start + np.timedelta64(1, "D")).astype("timedelta64[h]").astype(float)
    edges = [start + np.timedelta64(int(span * b / per_tile), "h") for b in range(per_tile + 1)]
    out = []
    for v in by_tile.values():
        # Preferred source first (Collection 1), then gap-filling L2A; each least cloudy first.
        v = sorted(v, key=lambda it: (not stac.is_c1(it), it.properties.get("eo:cloud_cover", 100)))
        chosen = []
        for b in range(per_tile):
            for it in v:
                d = np.datetime64(it.datetime.replace(tzinfo=None))
                if edges[b] <= d < edges[b + 1] and it not in chosen:
                    chosen.append(it)
                    break
        for it in v:
            if len(chosen) >= per_tile:
                break
            if it not in chosen:
                chosen.append(it)
        out.extend(chosen)
    return out


def composite_chunk(grid: Grid, year: int, per_tile: int = 6, max_cloud: float = 60.0, pool=None,
                    bands=stac.BANDS, balanced: bool = True, by_orbit: bool = False):
    """Median composite for one chunk.

    Returns (reflectance[bands, h, w] float32 NaN-masked, n_clear[h, w] int).
    """
    bbox = grid.lonlat_bbox(pad_deg=0.01)
    if balanced:
        items = select_scenes(stac.search_merged(bbox, year, max_cloud=max_cloud), per_tile, year=year,
                              by_orbit=by_orbit)
    else:
        items = select_scenes(stac.search(bbox, year, max_cloud=max_cloud), per_tile)
    nb = len(bands)
    if not items:
        return np.full((nb, grid.height, grid.width), np.nan, np.float32), np.zeros((grid.height, grid.width), int)
    own_pool = pool is None
    pool = pool or ThreadPoolExecutor(16)
    try:
        scl = list(pool.map(lambda it: read_to_grid(it.assets["scl"].href, grid, categorical=True), items))
        keep = []
        for it, s in zip(items, scl):
            if s is None:
                continue
            clear = np.isin(s, SCL_CLEAR)
            if clear.mean() > 0.02:
                keep.append((it, clear))
        jobs = [(k, j, it.assets[b].href) for k, (it, _) in enumerate(keep) for j, b in enumerate(bands)]
        arrs = list(pool.map(lambda job: read_to_grid(job[2], grid), jobs))
    finally:
        if own_pool:
            pool.shutdown()
    stack = np.full((len(keep), nb, grid.height, grid.width), np.nan, np.float32)
    for (k, j, _), a in zip(jobs, arrs):
        if a is not None:
            sc, off = stac.scale_offset(keep[k][0], bands[j])
            stack[k, j] = a * sc + off
    for k, (_, clear) in enumerate(keep):
        stack[k][:, ~clear] = np.nan
    # Reflectance outside [0, 1.2] after offset is a no-data or saturation artefact.
    stack[(stack < 0) | (stack > 1.2)] = np.nan
    n_clear = np.isfinite(stack[:, 0]).sum(0)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0) if len(keep) else np.full((nb, grid.height, grid.width), np.nan, np.float32)
    return med.astype(np.float32), n_clear


def composite_grid(grid: Grid, year: int, out_path: str, chunk: int = 1024, per_tile: int = 6,
                   max_cloud: float = 60.0, workers: int = 16, bands=stac.BANDS, overwrite: bool = False,
                   keep_chunk=None):
    """Write a uint16 GeoTIFF: bands scaled by 1e4 (0 = no data) + a clear-observation count band.

    `keep_chunk(sub_grid) -> bool` lets callers skip chunks (e.g. open sea or
    outside the country), which are left as no data.
    """
    if os.path.exists(out_path) and not overwrite:
        log.info("exists: %s", out_path)
        return out_path
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".part"
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=len(bands) + 1, dtype="uint16",
                   crs=grid.crs, transform=grid.transform, nodata=0, tiled=True, blockxsize=256,
                   blockysize=256, compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
    with ThreadPoolExecutor(workers) as pool, rasterio.open(tmp, "w", **profile) as dst:
        for i, b in enumerate(bands, 1):
            dst.set_band_description(i, b)
        dst.set_band_description(len(bands) + 1, "n_clear")
        chunks = [ch for ch in grid.chunks(chunk) if keep_chunk is None or keep_chunk(ch[2])]
        for n, (r, c, g) in enumerate(chunks, 1):
            med, cnt = composite_chunk(g, year, per_tile=per_tile, max_cloud=max_cloud, pool=pool, bands=bands)
            out = np.where(np.isfinite(med), np.clip(med * 1e4, 1, 65535), 0).astype(np.uint16)
            w = Window(c, r, g.width, g.height)
            dst.write(out, indexes=list(range(1, len(bands) + 1)), window=w)
            dst.write(np.clip(cnt, 0, 65535).astype(np.uint16), indexes=len(bands) + 1, window=w)
            log.info("%s year=%d chunk %d/%d clear-median=%.1f", os.path.basename(out_path), year, n, len(chunks),
                     float(np.median(cnt)))
    os.replace(tmp, out_path)
    return out_path


def fill_gaps(path: str, year: int, chunk: int = 1024, per_tile: int = 6, max_cloud: float = 60.0,
              workers: int = 16, min_px: int = 1000, bands=stac.BANDS) -> int:
    """Fill pixels with no clear observation, in place, using orbit-aware scene selection.

    Per-tile selection can, by chance, take all scenes of a tile from an orbit
    that does not reach part of it (e.g. eastern Khulna in 2026). Only pixels
    with n_clear == 0 inside processed chunks are replaced; all other pixels
    keep the original composite. Returns the number of filled pixels.
    """
    filled = 0
    with rasterio.open(path, "r+") as dst, ThreadPoolExecutor(workers) as pool:
        t = dst.transform
        grid = Grid(t.c, t.f, dst.width, dst.height, t.a)
        nb = len(bands)
        for r, c, g in grid.chunks(chunk):
            w = Window(c, r, g.width, g.height)
            cnt = dst.read(nb + 1, window=w)
            gap = cnt == 0
            if gap.sum() < min_px or gap.all():  # nothing to fill, or a chunk skipped on purpose
                continue
            med, new = composite_chunk(g, year, per_tile=per_tile, max_cloud=max_cloud, pool=pool, bands=bands,
                                       by_orbit=True)
            use = gap & (new > 0)
            if not use.any():
                continue
            old = dst.read(list(range(1, nb + 1)), window=w)
            fresh = np.where(np.isfinite(med), np.clip(med * 1e4, 1, 65535), 0).astype(np.uint16)
            old[:, use] = fresh[:, use]
            cnt[use] = np.clip(new[use], 0, 65535)
            dst.write(old, indexes=list(range(1, nb + 1)), window=w)
            dst.write(cnt, indexes=nb + 1, window=w)
            filled += int(use.sum())
            log.info("%s gap-fill chunk r=%d c=%d filled %d px", os.path.basename(path), r, c, int(use.sum()))
    return filled


def read_composite(path: str, window=None):
    """Load a composite as float32 reflectance (NaN = no data) plus the clear count."""
    with rasterio.open(path) as src:
        a = src.read(window=window).astype(np.float32)
    refl = a[:-1] / 1e4
    refl[a[:-1] == 0] = np.nan
    return refl, a[-1]
