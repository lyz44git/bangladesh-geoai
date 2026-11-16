"""Predicting local wealth from Sentinel-2 imagery (Bangladesh, ~2.4 km tiles).

Labels: Meta/UC Berkeley Relative Wealth Index (Chi et al. 2022, PNAS), 26,932
tiles. A second, independently produced label, Flowminder's 2011 DHS wealth-
index poverty map (Steele et al. 2017), checks that results are not specific
to one label-generating model.

Feature sets
  s2_stats   handcrafted per-tile statistics of bands and indices (2020 dry season, 40 m)
  mosaiks    1,024 random convolutional features from the same imagery
  landcover  ESA WorldCover 2021 class shares
  context    population (Meta HRSL) and modelled travel time to Dhaka (non-imagery)
Validation: 5-fold spatial-block CV (50 km blocks) vs. random CV; tile- and upazila-level R^2.
"""
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoai import labels  # noqa: E402
from geoai.composite import GDAL_ENV, read_composite  # noqa: E402
from geoai.features import indices  # noqa: E402
from geoai.grid import BD_TM, Grid, grid_from_lonlat, lonlat_to_tm  # noqa: E402
from geoai.mosaiks import RandomConvFeatures  # noqa: E402

CFG = json.loads((ROOT / "configs/aois.json").read_text())
DATA, TAB, FIG = ROOT / "data/wealth", ROOT / "outputs/tables", ROOT / "outputs/figures"
DATA.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
for _n in ("rasterio", "rasterio._env", "rasterio.env", "rasterio.session"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("wealth").info
HALF = 28  # pixels at 40 m -> 2.24 km tile (zoom-14 quadkey width at ~23.7N)
HRSL = "/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/hrsl_general/hrsl_general-latest.vrt"


def load_tiles():
    df = pd.read_csv(ROOT / "data/raw/rwi/bgd_relative_wealth_index.csv")
    x, y = lonlat_to_tm(df.longitude.values, df.latitude.values)
    df["x"], df["y"] = x, y
    up = gpd.read_file(ROOT / "data/raw/gadm/gadm41_BGD_3.json")[["GID_2", "NAME_1", "NAME_2", "GID_3", "NAME_3", "geometry"]]
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    j = gpd.sjoin(pts, up, how="left", predicate="within").drop(columns=["index_right", "geometry"])
    return j[~j.index.duplicated()].reset_index(drop=True)


def coarse_layers():
    """200 m national rasters: WorldCover shares and HRSL population (persons per cell)."""
    n = CFG["national"]
    g = grid_from_lonlat(n["lon_min"], n["lat_min"], n["lon_max"], n["lat_max"], 200)
    out = {}
    for name, cls in {"wc_built": 50, "wc_crop": 40, "wc_tree": 10, "wc_water": 80, "wc_wetland": 90}.items():
        p = DATA / f"{name}_200m.npy"
        if not p.exists():
            np.save(p, labels.class_fraction(g, cls, year=2021))
        out[name] = np.load(p)
    p = DATA / "pop_200m.npy"
    if not p.exists():
        dst = np.zeros((g.height, g.width), np.float32)
        with rasterio.Env(**{**GDAL_ENV, "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt"}):
            with rasterio.open(HRSL) as src:
                bb = g.lonlat_bbox(pad_deg=0.05)
                win = rasterio.windows.from_bounds(*bb, src.transform)
                factor = 4  # read at ~120 m, then average onto 200 m
                a = src.read(1, window=win, out_shape=(int(win.height // factor), int(win.width // factor)),
                             resampling=Resampling.average)
                a = np.nan_to_num(a, nan=0.0).astype(np.float32)
                tr = src.window_transform(win) * rasterio.Affine.scale(win.width / a.shape[1], win.height / a.shape[0])
                reproject(a, dst, src_transform=tr, src_crs=src.crs, dst_transform=g.transform, dst_crs=BD_TM,
                          resampling=Resampling.average)
                # convert mean persons per ~30 m source pixel to persons per 200 m cell
                src_px_m2 = (abs(src.res[0]) * 111_320 * np.cos(np.deg2rad(23.7))) * (abs(src.res[1]) * 110_570)
                dst *= (200 * 200) / src_px_m2
        np.save(p, dst)
    out["pop"] = np.load(p)
    return g, out


def tile_window_stats(g: Grid, arr, x, y, half_m=1120, how="mean"):
    r0 = ((g.ymax - (y + half_m)) / g.res).astype(int)
    r1 = ((g.ymax - (y - half_m)) / g.res).astype(int)
    c0 = ((x - half_m - g.xmin) / g.res).astype(int)
    c1 = ((x + half_m - g.xmin) / g.res).astype(int)
    vals = np.full(len(x), np.nan)
    for i in range(len(x)):
        sub = arr[max(r0[i], 0):max(r1[i], 0), max(c0[i], 0):max(c1[i], 0)]
        if sub.size:
            vals[i] = np.nansum(sub) if how == "sum" else np.nanmean(sub)
    return vals


def sample_raster(path, x, y, crs=None):
    with rasterio.open(path) as src:
        if crs is None and src.crs != BD_TM:
            from pyproj import Transformer

            t = Transformer.from_crs(BD_TM, src.crs, always_xy=True)
            xx, yy = t.transform(x, y)
        else:
            xx, yy = x, y
        v = np.array([s[0] for s in src.sample(zip(xx, yy))], dtype=float)
        if src.nodata is not None:
            v[v == src.nodata] = np.nan
    return v


def build_features(df):
    feat_path = DATA / "features.parquet"
    if feat_path.exists():
        return pd.read_parquet(feat_path), np.load(DATA / "mosaiks.npy")
    comp = str(ROOT / "data/composites/national_2020.tif")
    with rasterio.open(comp) as src:
        t = src.transform
        g = Grid(t.c, t.f, src.width, src.height, t.a)
    rows = ((g.ymax - df.y.values) / g.res).astype(int)
    cols = ((df.x.values - g.xmin) / g.res).astype(int)
    imgs = np.full((len(df), 6, 2 * HALF, 2 * HALF), np.nan, np.float16)
    stats = []
    names = None
    for i, (r, c) in enumerate(zip(rows, cols)):
        win = rasterio.windows.Window(c - HALF, r - HALF, 2 * HALF, 2 * HALF)
        try:
            refl, n_clear = read_composite(comp, window=win)
        except Exception:
            refl = np.full((6, 2 * HALF, 2 * HALF), np.nan, np.float32)
        if refl.shape[1:] != (2 * HALF, 2 * HALF):
            refl = np.full((6, 2 * HALF, 2 * HALF), np.nan, np.float32)
        imgs[i] = refl
        idx = indices(refl)
        layers = {**{b: refl[k] for k, b in enumerate(["blue", "green", "red", "nir", "swir16", "swir22"])}, **idx}
        s = {}
        for k, v in layers.items():
            v = v[np.isfinite(v)]
            if v.size < 50:
                s.update({f"{k}_{q}": np.nan for q in ("mean", "sd", "p10", "p50", "p90")})
                continue
            s.update({f"{k}_mean": v.mean(), f"{k}_sd": v.std(), f"{k}_p10": np.percentile(v, 10),
                      f"{k}_p50": np.percentile(v, 50), f"{k}_p90": np.percentile(v, 90)})
        nd = idx["ndbi"]
        s["share_ndbi_pos"] = float(np.nanmean(nd > 0)) if np.isfinite(nd).any() else np.nan
        s["valid_frac"] = float(np.isfinite(refl).all(0).mean())
        stats.append(s)
        if i % 5000 == 0:
            log(f"tile features {i}/{len(df)}")
    S = pd.DataFrame(stats)
    S.columns = ["s2_" + c for c in S.columns]
    # MOSAIKS random features
    ok = S.s2_valid_frac.values > 0.8
    ok_idx = np.flatnonzero(ok)
    rcf = RandomConvFeatures(n_filters=512, ksize=3, seed=0).fit(imgs[ok_idx[::5]].astype(np.float32))
    M = np.full((len(df), 1024), np.nan, np.float32)
    for s in range(0, len(ok_idx), 4096):  # batches keep peak memory low (images stay float16)
        j = ok_idx[s:s + 4096]
        M[j] = rcf.transform(imgs[j])
    del imgs
    # context + land cover
    cg, layers = coarse_layers()
    C = pd.DataFrame(index=df.index)
    for k in ("wc_built", "wc_crop", "wc_tree", "wc_water", "wc_wetland"):
        C["lc_" + k[3:]] = tile_window_stats(cg, layers[k], df.x.values, df.y.values)
    C["ctx_log_pop"] = np.log1p(tile_window_stats(cg, layers["pop"], df.x.values, df.y.values, how="sum"))
    C["ctx_log_tt_dhaka"] = np.log(sample_raster(ROOT / "data/travel/tt_post.tif", df.x.values, df.y.values) + 0.1)
    fm = sample_raster(ROOT / "data/raw/flowminder/bgd2011wipov.tif", df.longitude.values, df.latitude.values, crs="ll")
    out = pd.concat([df.reset_index(drop=True), S, C], axis=1)
    out["flowminder_wipov2011"] = fm
    out.to_parquet(feat_path)
    np.save(DATA / "mosaiks.npy", M)
    return out, M


def spatial_folds(df, block_km=50, k=5, seed=0):
    b = (np.floor(df.x / (block_km * 1000)).astype(int) * 1000 + np.floor(df.y / (block_km * 1000)).astype(int)).values
    ub = np.unique(b)
    rng = np.random.default_rng(seed)
    assign = dict(zip(ub, rng.permutation(len(ub)) % k))
    return np.array([assign[v] for v in b])


def cv_predict(X, y, folds, kind):
    pred = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if kind == "ridge":
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
        else:
            m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                                  verbose=-1, n_jobs=4)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def scores(y, p, df):
    ok = np.isfinite(y) & np.isfinite(p)
    d = df.loc[ok].assign(y=y[ok], p=p[ok], w=np.exp(df.loc[ok, "ctx_log_pop"]) - 1)
    agg = d.groupby("GID_3").apply(lambda g: pd.Series({"y": np.average(g.y, weights=g.w + 1e-6),
                                                         "p": np.average(g.p, weights=g.w + 1e-6), "n": len(g)}),
                                   include_groups=False)
    agg = agg[agg.n >= 3]
    return dict(n=int(ok.sum()), r2=r2_score(y[ok], p[ok]), spearman=spearmanr(y[ok], p[ok]).statistic,
                r2_upazila=r2_score(agg.y, agg.p), n_upazila=len(agg))


def main():
    df = load_tiles()
    log(f"{len(df)} RWI tiles")
    D, M = build_features(df)
    keep = (D.s2_valid_frac > 0.8).values & np.isfinite(M).all(1)
    D, M = D[keep].reset_index(drop=True), M[keep]
    log(f"{len(D)} tiles with valid imagery")
    s2 = [c for c in D.columns if c.startswith("s2_") and c != "s2_valid_frac"]
    lc = [c for c in D.columns if c.startswith("lc_")]
    ctx = ["ctx_log_pop", "ctx_log_tt_dhaka"]
    pcs = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(M))
    sets = {
        "mosaiks (ridge)": ("ridge", M),
        "s2_stats (gbm)": ("gbm", D[s2].values),
        "landcover (gbm)": ("gbm", D[lc].values),
        "s2_stats+landcover (gbm)": ("gbm", D[s2 + lc].values),
        "imagery all: s2+lc+mosaiks50 (gbm)": ("gbm", np.column_stack([D[s2 + lc].values, pcs])),
        "context only: pop+travel time (gbm)": ("gbm", D[ctx].values),
        "imagery + context (gbm)": ("gbm", np.column_stack([D[s2 + lc + ctx].values, pcs])),
    }
    folds_sp = spatial_folds(D)
    folds_rd = np.random.default_rng(0).permutation(len(D)) % 5
    rows, preds = [], {}
    for target in ("rwi", "flowminder_wipov2011"):
        y = D[target].values.astype(float)
        for name, (kind, X) in sets.items():
            for cvname, folds in (("spatial 50km", folds_sp), ("random", folds_rd)):
                if cvname == "random" and target != "rwi":
                    continue
                ok = np.isfinite(y)
                p = np.full(len(y), np.nan)
                p[ok] = cv_predict(X[ok], y[ok], folds[ok], kind)
                rows.append(dict(target=target, features=name, cv=cvname, **scores(y, p, D)))
                log(f"{target:22s} {name:38s} {cvname:12s} R2={rows[-1]['r2']:.3f} upazila R2={rows[-1]['r2_upazila']:.3f}")
                if cvname == "spatial 50km":
                    preds[(target, name)] = p
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "wealth_prediction_results.csv", index=False)
    best = "imagery all: s2+lc+mosaiks50 (gbm)"
    out = D[["latitude", "longitude", "rwi", "flowminder_wipov2011", "GID_2", "NAME_2", "GID_3", "NAME_3",
             "ctx_log_pop", "lc_built"]].copy()
    out["rwi_pred_imagery"] = preds[("rwi", best)]
    out["rwi_pred_imagery_context"] = preds[("rwi", "imagery + context (gbm)")]
    out.to_parquet(DATA / "predictions.parquet")
    # Label agreement between the two independently modelled wealth maps.
    ok = np.isfinite(D.flowminder_wipov2011)
    agree = dict(spearman_rwi_vs_flowminder_poverty=float(spearmanr(D.rwi[ok], D.flowminder_wipov2011[ok]).statistic),
                 n=int(ok.sum()))
    (TAB / "wealth_label_agreement.json").write_text(json.dumps(agree, indent=2))
    # Feature importance of the best imagery model fit on all data.
    X = np.column_stack([D[s2 + lc].values, pcs])
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=4).fit(X, D.rwi.values)
    imp = pd.DataFrame({"feature": s2 + lc + [f"mosaiks_pc{i + 1}" for i in range(50)],
                        "gain": m.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp["gain_share"] = imp.gain / imp.gain.sum()
    imp.to_csv(TAB / "wealth_feature_importance.csv", index=False)
    plot(out, res)
    log("done")


def plot(out, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    kw = dict(s=1.0, cmap="RdYlBu", vmin=-1.0, vmax=1.5)
    ax[0].scatter(out.longitude, out.latitude, c=out.rwi, **kw)
    ax[0].set_title("Relative Wealth Index (Meta, label)")
    sc = ax[1].scatter(out.longitude, out.latitude, c=out.rwi_pred_imagery, **kw)
    ax[1].set_title("Predicted from Sentinel-2 only\n(spatial-block out-of-fold)")
    plt.colorbar(sc, ax=ax[1], fraction=0.04)
    resid = out.rwi_pred_imagery - out.rwi
    sc = ax[2].scatter(out.longitude, out.latitude, c=resid, s=1.0, cmap="PuOr", vmin=-1, vmax=1)
    ax[2].set_title("Residual (predicted - label)")
    plt.colorbar(sc, ax=ax[2], fraction=0.04)
    for a in ax[:3]:
        a.set_aspect(1 / np.cos(np.deg2rad(23.7)))
        a.set_xticks([]), a.set_yticks([])
    r = res[(res.target == "rwi")].pivot(index="features", columns="cv", values="r2")
    r = r.sort_values("spatial 50km")
    yy = np.arange(len(r))
    ax[3].barh(yy - 0.2, r["spatial 50km"], height=0.4, label="spatial-block CV (50 km)")
    ax[3].barh(yy + 0.2, r["random"], height=0.4, label="random CV", alpha=0.6)
    ax[3].set_yticks(yy, r.index, fontsize=8)
    ax[3].set_xlabel("out-of-fold R$^2$ (tile level)")
    ax[3].legend(fontsize=8)
    ax[3].set_title("RWI prediction accuracy by feature set")
    fig.tight_layout()
    fig.savefig(FIG / "wealth_maps_and_accuracy.png", dpi=110)


if __name__ == "__main__":
    main()
