"""Travel time to Dhaka with and without the Padma Bridge (500 m friction surface).

Outputs data/travel/tt_{scenario}.tif (hours) and outputs/tables/padma_bridge_segments.csv.
Scenarios:
  post        current Overture network (bridge open, ferries kept)
  pre         same network minus the Padma Bridge road segments
  pre_slow    pre with slower ferries (1.5 km/h effective)  -- sensitivity
  pre_fast    pre with faster ferries (4 km/h effective)    -- sensitivity
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import rasterio
import rasterio.features
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoai import labels, travel  # noqa: E402
from geoai.grid import BD_TM, grid_from_lonlat  # noqa: E402

CFG = json.loads((ROOT / "configs/aois.json").read_text())
OUT = ROOT / "data/travel"
OUT.mkdir(parents=True, exist_ok=True)
DHAKA = (90.40, 23.78)  # Dhaka city centre (approx.)
BRIDGE_BOX = shapely.box(90.20, 23.38, 90.33, 23.50)


def load_network():
    t = pq.read_table(ROOT / "data/raw/overture/bgd_segments_major.parquet").to_pandas()
    g = gpd.GeoDataFrame(t.drop(columns=["geometry", "bbox"]), geometry=shapely.from_wkb(t.geometry), crs=4326)
    return g.to_crs(BD_TM)


def frictions(ferry_speeds=(2.0, 1.5, 4.0)):
    """National 500 m friction surfaces with and without the bridge.

    Returns (grid, inside, {scenario: friction}) with scenarios as in the module docstring.
    """
    n = CFG["national"]
    grid = grid_from_lonlat(n["lon_min"], n["lat_min"], n["lon_max"], n["lat_max"], 500)
    water_path = OUT / "water_frac_500m.npy"
    if water_path.exists():
        water = np.load(water_path)
    else:
        water = labels.class_fraction(grid, labels.WATER, year=2021)
        np.save(water_path, water)

    # Outside Bangladesh is impassable: cross-border trips are not a realistic route.
    bd = gpd.read_file(ROOT / "data/raw/gadm/gadm41_BGD_0.json").to_crs(BD_TM)
    inside = rasterio.features.rasterize(((geom.buffer(1000), 1) for geom in bd.geometry),
                                         out_shape=(grid.height, grid.width), transform=grid.transform, dtype="uint8")
    water_eff = np.where(inside == 1, water, 1.0)

    net = load_network()
    roads = net[net.subtype == "road"]
    ferries = net[net.subtype == "water"]
    box_tm = gpd.GeoSeries([BRIDGE_BOX], crs=4326).to_crs(BD_TM).iloc[0]
    cand = roads[(roads["class"] == "motorway") & roads.intersects(box_tm)].copy()
    cand["len_km"] = cand.length / 1000
    bridge = cand[cand.len_km > 8.0]
    bridge.drop(columns="geometry").assign(
        centroid_lonlat=gpd.GeoSeries(bridge.centroid, crs=BD_TM).to_crs(4326).astype(str)
    ).to_csv(ROOT / "outputs/tables/padma_bridge_segments.csv", index=False)
    print(f"bridge segments: {len(bridge)} ({bridge.len_km.round(2).tolist()} km)")
    roads_pre = roads.drop(index=bridge.index)
    tag = {2.0: "", 1.5: "_slow", 4.0: "_fast"}
    fr = {}
    for fs in ferry_speeds:
        fr["post" + tag[fs]] = travel.friction_surface(grid, roads, water_eff, ferries=ferries, ferry_speed=fs)
        fr["pre" + tag[fs]] = travel.friction_surface(grid, roads_pre, water_eff, ferries=ferries, ferry_speed=fs)
    return grid, inside, fr


def main():
    grid, inside, fr = frictions()
    prof = dict(driver="GTiff", width=grid.width, height=grid.height, count=1, dtype="float32", crs=BD_TM,
                transform=grid.transform, nodata=np.nan, compress="deflate")
    for name, f in fr.items():
        tt = travel.travel_time_to(grid, f, *DHAKA).astype(np.float32)
        tt[(inside == 0) | ~np.isfinite(tt)] = np.nan
        with rasterio.open(OUT / f"tt_{name}.tif", "w", **prof) as dst:
            dst.write(tt, 1)
        print(name, "median hours (inside BD):", float(np.nanmedian(tt)))


if __name__ == "__main__":
    main()
