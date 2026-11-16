"""Build annual dry-season Sentinel-2 composites for every study area.

  python scripts/01_build_composites.py cities    # 6 cities x 9 years, 20 m
  python scripts/01_build_composites.py padma     # Padma study region x 9 years, 30 m
  python scripts/01_build_composites.py national  # Bangladesh 2020 season, 40 m (wealth study)
"""
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoai import composite  # noqa: E402
from geoai.grid import BD_TM, grid_around, grid_from_lonlat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/aois.json").read_text())
OUT = ROOT / "data/composites"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
for _n in ("rasterio", "rasterio._env", "rasterio.env", "rasterio.session"):
    logging.getLogger(_n).setLevel(logging.WARNING)


def country_filter(buffer_km=5):
    bd = gpd.read_file(ROOT / "data/raw/gadm/gadm41_BGD_0.json").to_crs(BD_TM).geometry.union_all()
    bd = bd.buffer(buffer_km * 1000)
    return lambda g: bd.intersects(box(*g.bounds))


def main(which):
    years = CFG["years"]
    if which == "cities":
        for y in years:
            for name, c in CFG["cities"].items():
                g = grid_around(c["lon"], c["lat"], c["half_km"], CFG["city_res_m"])
                composite.composite_grid(g, y, str(OUT / f"city_{name}_{y}.tif"), workers=24)
    elif which == "padma":
        p = CFG["padma_region"]
        g = grid_from_lonlat(p["lon_min"], p["lat_min"], p["lon_max"], p["lat_max"], CFG["padma_res_m"])
        keep = country_filter()
        order = [2021] + [y for y in years if y != 2021]  # training year first
        for y in order:
            composite.composite_grid(g, y, str(OUT / f"padma_{y}.tif"), chunk=1024, per_tile=5, workers=24,
                                     keep_chunk=keep)
    elif which == "national":
        n = CFG["national"]
        g = grid_from_lonlat(n["lon_min"], n["lat_min"], n["lon_max"], n["lat_max"], CFG["national_res_m"])
        composite.composite_grid(g, 2020, str(OUT / "national_2020.tif"), chunk=1024, per_tile=5, workers=24,
                                 keep_chunk=country_filter(buffer_km=3))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1])
