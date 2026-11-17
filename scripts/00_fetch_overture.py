"""Extract Bangladesh major roads, railways and ferry routes from Overture Maps (S3, anonymous)."""
import os
import sys
import time
from urllib.parse import urlparse

import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

RELEASE = sys.argv[1] if len(sys.argv) > 1 else "2026-10-21.0"
OUT = "data/raw/overture/bgd_segments_major.parquet"
_kw = {}
if os.environ.get("HTTPS_PROXY"):  # use a proxy only when one is configured
    proxy = urlparse(os.environ["HTTPS_PROXY"])
    _kw["proxy_options"] = {"scheme": "http", "host": proxy.hostname, "port": proxy.port}
s3 = pafs.S3FileSystem(anonymous=True, region="us-west-2", **_kw)
path = f"overturemaps-us-west-2/release/{RELEASE}/theme=transportation/type=segment/"
t = time.time()
dset = ds.dataset(path, filesystem=s3, format="parquet")
f = ds.field
flt = ((f("bbox", "xmin") > 88.0) & (f("bbox", "xmax") < 92.8) & (f("bbox", "ymin") > 20.5) & (f("bbox", "ymax") < 26.7)
       & (f("subtype").isin(["water", "rail"]) | f("class").isin(["motorway", "trunk", "primary", "secondary", "tertiary"])))
cols = ["id", "subtype", "class", "geometry", "bbox"]
tbl = dset.to_table(columns=cols, filter=flt)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pq.write_table(tbl, OUT)
print(f"{tbl.num_rows} segments in {time.time() - t:.0f}s")
