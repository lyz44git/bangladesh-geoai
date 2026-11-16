"""Raster travel-time model (least-cost path over a friction surface).

Travel time to a destination is the minimum, over all paths, of the sum of
cell traversal times. Roads set the speed on the cells they cross; open land
is walkable; large water bodies are impassable except where a road (bridge)
or a ferry route crosses them. Counterfactual networks (e.g. "no bridge") are
built by removing segments before rasterising.
"""
from __future__ import annotations

import numpy as np
from rasterio.features import rasterize
from skimage.graph import MCP_Geometric

from .grid import Grid, lonlat_to_tm

# Speeds in km/h. Deliberately conservative for Bangladesh's congested,
# mixed-traffic roads; results are reported as log changes, which are robust
# to proportional rescaling of all speeds.
ROAD_SPEED = {"motorway": 70.0, "trunk": 50.0, "primary": 40.0, "secondary": 30.0, "tertiary": 25.0}
OFFROAD_SPEED = 5.0


def friction_surface(grid: Grid, roads, water_frac: np.ndarray, ferries=None, ferry_speed: float = 4.0,
                     water_thr: float = 0.5) -> np.ndarray:
    """Seconds needed to cross one cell (straight). Inf = impassable.

    roads: GeoDataFrame in grid CRS with a `class` column.
    ferries: GeoDataFrame of ferry routes; `ferry_speed` is an *effective*
    speed that folds waiting/loading time into the crossing.
    """
    speed = np.full((grid.height, grid.width), OFFROAD_SPEED, np.float32)
    speed[np.nan_to_num(water_frac, nan=1.0) >= water_thr] = 0.0
    if ferries is not None and len(ferries):
        f = rasterize(((geom, 1) for geom in ferries.geometry), out_shape=speed.shape, transform=grid.transform,
                      all_touched=True, dtype="uint8")
        speed = np.where(f == 1, np.maximum(speed, ferry_speed), speed)
    # Slowest classes first so faster roads overwrite them where they overlap.
    for cls in sorted(ROAD_SPEED, key=ROAD_SPEED.get):
        sub = roads[roads["class"] == cls]
        if not len(sub):
            continue
        r = rasterize(((geom, 1) for geom in sub.geometry), out_shape=speed.shape, transform=grid.transform,
                      all_touched=True, dtype="uint8")
        speed = np.where(r == 1, np.maximum(speed, ROAD_SPEED[cls]), speed)
    with np.errstate(divide="ignore"):
        sec = grid.res / (speed / 3.6)
    return sec.astype(np.float64)


def travel_time_to(grid: Grid, friction: np.ndarray, lon: float, lat: float, radius_cells: int = 1) -> np.ndarray:
    """Hours from every cell to the destination (cells within radius count as the destination)."""
    x, y = lonlat_to_tm(lon, lat)
    r0 = int((grid.ymax - y) / grid.res)
    c0 = int((x - grid.xmin) / grid.res)
    starts = [(r0 + dr, c0 + dc) for dr in range(-radius_cells, radius_cells + 1)
              for dc in range(-radius_cells, radius_cells + 1)]
    cost = np.where(np.isfinite(friction), friction, np.inf)
    # MCP_Geometric weights diagonal moves by sqrt(2); costs are per-cell, so
    # the accumulated value is the traversal time in seconds.
    mcp = MCP_Geometric(cost, fully_connected=True)
    acc, _ = mcp.find_costs(starts)
    return acc / 3600.0
