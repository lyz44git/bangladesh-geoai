"""Sentinel-2 L2A scene search on the Earth Search STAC API (AWS open data)."""
from __future__ import annotations

import functools
import re
import time
import warnings

from pystac_client import Client

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"

# Collection 1 is reprocessed with one processing baseline (05.00+) and is the
# preferred source for time series. Its archive is sparse before late 2019,
# so earlier seasons fall back to the original L2A collection.
C1 = "sentinel-2-c1-l2a"
L2A = "sentinel-2-l2a"

BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]


def dry_season(year: int) -> tuple[str, str]:
    """Bangladesh dry season ending in `year`: 15 Nov (year-1) to 31 Mar (year).

    Cloud-free scenes are scarce during the June-October monsoon, so annual
    composites are built from the dry season only.
    """
    return f"{year - 1}-11-15", f"{year}-03-31"


def collection_for(year: int) -> str:
    return C1 if year >= 2020 else L2A


@functools.lru_cache(maxsize=1)
def _client() -> Client:
    return Client.open(EARTH_SEARCH)


def search(bbox_lonlat, year: int, max_cloud: float = 60.0, collection: str | None = None, retries: int = 4):
    """Return STAC items intersecting bbox in the dry season, least cloudy first."""
    start, end = dry_season(year)
    coll = collection or collection_for(year)
    for attempt in range(retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                items = list(
                    _client()
                    .search(
                        collections=[coll],
                        bbox=list(bbox_lonlat),
                        datetime=f"{start}/{end}",
                        query={"eo:cloud_cover": {"lt": max_cloud}},
                        max_items=2000,
                    )
                    .items()
                )
            break
        except Exception:  # transient API errors
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return items


def search_merged(bbox_lonlat, year: int, max_cloud: float = 60.0):
    """Collection 1 scenes plus original-L2A scenes that fill C1 archive gaps.

    The C1 archive has holes (e.g. almost nothing between mid-December 2021 and
    late March 2022 over parts of Bangladesh). Original L2A items for the same
    acquisitions are appended after the C1 items; scene selection prefers C1 and
    falls back to L2A only for date windows C1 does not cover.
    """
    if year < 2020:
        return search(bbox_lonlat, year, max_cloud, collection=L2A)
    c1 = search(bbox_lonlat, year, max_cloud, collection=C1)
    seen = {(tile_id(it), it.datetime.date()) for it in c1}
    extra = [it for it in search(bbox_lonlat, year, max_cloud, collection=L2A)
             if (tile_id(it), it.datetime.date()) not in seen]
    return c1 + extra


def is_c1(item) -> bool:
    return item.collection_id == C1


def scale_offset(item, band: str) -> tuple[float, float]:
    """Reflectance = DN * scale + offset, from the asset's raster:bands metadata.

    Collection 1 assets carry offset -0.1 (the 05.00 BOA_ADD_OFFSET). Original
    L2A items on Earth Search already have the offset removed
    (earthsearch:boa_offset_applied) or predate it, so they default to 1e-4, 0.
    """
    rb = item.assets[band].extra_fields.get("raster:bands") or [{}]
    return float(rb[0].get("scale", 1e-4)), float(rb[0].get("offset", 0.0))


def rel_orbit(item) -> str:
    """Relative orbit (e.g. 'R033') from the product URI; '' when unknown.

    Some MGRS tiles are imaged from two neighbouring orbits, each covering only
    part of the tile, so scenes of one tile can have disjoint footprints.
    """
    m = re.search(r"_R(\d{3})_", item.properties.get("s2:product_uri", ""))
    return f"R{m.group(1)}" if m else ""


def tile_id(item) -> str:
    p = item.properties
    return p.get("grid:code") or p.get("s2:mgrs_tile") or item.id.split("_")[1]
