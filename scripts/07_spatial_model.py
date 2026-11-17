"""Module 5: a quantitative spatial model of Bangladesh's upazilas and the Padma Bridge.

Stages
  times   bilateral upazila-to-upazila travel times with and without the bridge
          (500 m friction surfaces from 04_travel_time.py; cached in data/qse/)
  model   calibration, counterfactuals, Bayesian uncertainty, link to module 4

Calibration data (all pre-opening): HRSL population, Meta Relative Wealth
Index (mapped to wages), WorldCover built-up share. See src/geoai/qse.py for
the model.

Outputs
  outputs/tables/qse_calibration.json
  outputs/tables/qse_padma_main.csv          upazila-level results, main parameters
  outputs/tables/qse_padma_sensitivity.csv   parameter grid
  outputs/tables/qse_padma_bayes.json        prior vs posterior (conditioned on the module-4 estimate)
  outputs/figures/qse_padma.png
"""
import importlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from geoai import qse  # noqa: E402
from geoai.grid import BD_TM, grid_from_lonlat  # noqa: E402

CFG = json.loads((ROOT / "configs/aois.json").read_text())
DATA = ROOT / "data/qse"
DATA.mkdir(parents=True, exist_ok=True)
TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"
SCEN = ["pre", "post", "pre_slow", "post_slow", "pre_fast", "post_fast"]
BD_POP_2022 = 169.83e6  # 2022 census total; HRSL is only used for relative shares

# Main parameters (literature values) and the module-4 estimate the model is confronted with.
# alpha_N: spending share on local non-traded goods and services (0 = one-sector model).
MAIN = dict(sigma=5.0, alpha=0.75, alpha_N=0.30, kappa=1.0, eps=3.0, wage_sd=0.25)
# Copied from outputs/tables/padma_did_results.csv: coefficient of specification (2) and its district-clustered
# standard error (the conservative choice). Update both if 06_padma_causal.py is re-run.
RF_COEF, RF_SE = 0.0596, 0.089


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ units
def units():
    """Upazila raster on the national 1 km grid, population, origins and covariates."""
    import geopandas as gpd
    import rasterio.features

    out = DATA / "units.parquet"
    ids_path = DATA / "ids_1km.npy"
    n = CFG["national"]
    g1 = grid_from_lonlat(n["lon_min"], n["lat_min"], n["lon_max"], n["lat_max"], 1000)
    if out.exists() and ids_path.exists():
        return pd.read_parquet(out), np.load(ids_path), g1
    up = gpd.read_file(ROOT / "data/raw/gadm/gadm41_BGD_3.json").to_crs(BD_TM).reset_index(drop=True)
    ids = rasterio.features.rasterize(((geom, i + 1) for i, geom in enumerate(up.geometry)),
                                      out_shape=(g1.height, g1.width), transform=g1.transform, dtype="int32")
    built = np.load(ROOT / "data/wealth/wc_built_200m.npy")
    H, W = g1.height, g1.width
    pop1 = hrsl_1km(g1)
    built1 = np.nanmean(built[: H * 5, : W * 5].reshape(H, 5, W, 5), axis=(1, 3))
    k = len(up)
    m = ids > 0
    cnt = np.bincount(ids[m], minlength=k + 1)[1:]
    P = np.bincount(ids[m], weights=pop1[m], minlength=k + 1)[1:]
    B = np.bincount(ids[m], weights=np.nan_to_num(built1[m]), minlength=k + 1)[1:]
    rr, cc = np.nonzero(m)
    # origin = most populous 1 km cell of the upazila (its main town)
    order = np.lexsort((-pop1[rr, cc], ids[rr, cc]))
    first = np.unique(ids[rr, cc][order], return_index=True)[1]
    orig = {int(ids[rr[order][f], cc[order][f]]): (int(rr[order][f]), int(cc[order][f])) for f in first}
    df = pd.DataFrame({"uid": np.arange(1, k + 1), "GID_3": up.GID_3, "NAME_1": up.NAME_1, "NAME_2": up.NAME_2,
                       "NAME_3": up.NAME_3, "cells_1km": cnt, "pop_raw": P, "built_share": B / np.maximum(cnt, 1)})
    df["orig_r1"] = df.uid.map(lambda u: orig.get(u, (-1, -1))[0])
    df["orig_c1"] = df.uid.map(lambda u: orig.get(u, (-1, -1))[1])
    df["x"] = g1.xmin + (df.orig_c1 + 0.5) * 1000
    df["y"] = g1.ymax - (df.orig_r1 + 0.5) * 1000
    df = df[(df.cells_1km > 0) & (df.pop_raw > 0)].reset_index(drop=True)
    df["pop"] = df.pop_raw / df.pop_raw.sum() * BD_POP_2022
    df["area_km2"] = df.cells_1km.astype(float)
    df = df.join(rwi_by_upazila(df, ids, pop1, g1), on="uid")
    df.to_parquet(out)
    np.save(ids_path, ids)
    np.save(DATA / "pop_1km.npy", pop1)
    return df, ids, g1


HRSL = "/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/hrsl_general/hrsl_general-latest.vrt"


def hrsl_1km(g1, tile_deg=0.5, workers=8):
    """HRSL population summed from native 1-arcsec pixels onto the national 1 km grid.

    HRSL is no data where no buildings were detected, and its overviews average
    over settled pixels only, so reading overviews and rescaling inflates sparse
    rural areas relative to dense cities. Summing native pixels avoids that.
    """
    import rasterio
    from concurrent.futures import ThreadPoolExecutor

    from geoai.composite import GDAL_ENV
    from geoai.grid import lonlat_to_tm

    path = DATA / "pop_hrsl_1km.npy"
    if path.exists():
        return np.load(path)
    lon0, lat0, lon1, lat1 = g1.lonlat_bbox(pad_deg=0.02)
    tiles = [(x, y) for x in np.arange(lon0, lon1, tile_deg) for y in np.arange(lat0, lat1, tile_deg)]
    env = {**GDAL_ENV, "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt"}

    def one(xy):
        x, y = xy
        with rasterio.Env(**env), rasterio.open(HRSL) as src:
            win = rasterio.windows.from_bounds(x, y, x + tile_deg, y + tile_deg, src.transform).round_offsets().round_lengths()
            a = src.read(1, window=win)
            tr = src.window_transform(win)
        a = np.where(np.isfinite(a) & (a > 0), a, 0.0)
        h, w = (a.shape[0] // 30) * 30, (a.shape[1] // 30) * 30
        b = a[:h, :w].reshape(h // 30, 30, w // 30, 30).sum((1, 3))
        r, c = np.nonzero(b > 0)
        lon = tr.c + (c * 30 + 15) * tr.a
        lat = tr.f + (r * 30 + 15) * tr.e
        return lon, lat, b[r, c], float(a.sum())

    out = np.zeros((g1.height, g1.width))
    with ThreadPoolExecutor(workers) as ex:
        for k, (lon, lat, v, _) in enumerate(ex.map(one, tiles)):
            if len(v):
                tx, ty = lonlat_to_tm(lon, lat)
                rr = ((g1.ymax - ty) // 1000).astype(int)
                cc = ((tx - g1.xmin) // 1000).astype(int)
                ok = (rr >= 0) & (rr < g1.height) & (cc >= 0) & (cc < g1.width)
                np.add.at(out, (rr[ok], cc[ok]), v[ok])
            if (k + 1) % 20 == 0:
                log(f"HRSL tiles {k + 1}/{len(tiles)}")
    np.save(path, out)
    return out


def rwi_by_upazila(df, ids, pop1, g1):
    """Population-weighted mean Meta RWI per upazila (nearest point for units without one)."""
    from geoai.grid import lonlat_to_tm

    r = pd.read_csv(ROOT / "data/raw/rwi/bgd_relative_wealth_index.csv")
    x, y = lonlat_to_tm(r.longitude.values, r.latitude.values)
    row = ((g1.ymax - y) // 1000).astype(int)
    col = ((x - g1.xmin) // 1000).astype(int)
    ok = (row >= 0) & (row < g1.height) & (col >= 0) & (col < g1.width)
    r, row, col, x, y = r[ok], row[ok], col[ok], x[ok], y[ok]
    r = r.assign(uid=ids[row, col], wt=pop1[row, col] + 1.0, x=x, y=y)
    g = r[r.uid > 0].groupby("uid").apply(lambda d: np.average(d.rwi, weights=d.wt), include_groups=False)
    out = pd.Series(index=df.uid.values, dtype=float)
    out.loc[g.index.intersection(out.index)] = g
    miss = out.index[out.isna()]
    if len(miss):
        sub = df.set_index("uid").loc[miss]
        d2 = (sub.x.values[:, None] - r.x.values[None]) ** 2 + (sub.y.values[:, None] - r.y.values[None]) ** 2
        out.loc[miss] = r.rwi.values[d2.argmin(1)]
    return out.rename("rwi")


# ------------------------------------------------------------------ travel times
_G = {}


def _init(fr, ids, pop1, k):
    from skimage.graph import MCP_Geometric

    _G.update(cost=np.where(np.isfinite(fr), fr, np.inf), ids=ids, pop1=pop1, k=k, MCP=MCP_Geometric)


def _row(start):
    """Population-weighted mean hours from one origin to every upazila (1 km aggregation)."""
    acc, _ = _G["MCP"](_G["cost"], fully_connected=True).find_costs([start])
    H, W = _G["ids"].shape
    T = acc[: 2 * H, : 2 * W].reshape(H, 2, W, 2) / 3600.0
    T = np.where(np.isfinite(T), T, np.nan)
    with np.errstate(all="ignore"):
        T1 = np.nanmean(T, axis=(1, 3))
    ids, w = _G["ids"], _G["pop1"]
    m = (ids > 0) & np.isfinite(T1) & (w > 0)
    num = np.bincount(ids[m], weights=w[m] * T1[m], minlength=_G["k"] + 1)[1:]
    den = np.bincount(ids[m], weights=w[m], minlength=_G["k"] + 1)[1:]
    with np.errstate(all="ignore"):
        return (num / den).astype(np.float32)


def stage_times():
    tt = importlib.import_module("04_travel_time")
    df, ids, g1 = units()
    pop1 = np.load(DATA / "pop_1km.npy")
    grid, inside, frs = tt.frictions()
    k = int(ids.max())
    for name in SCEN:
        out = DATA / f"T_{name}.npy"
        if out.exists():
            continue
        fr = frs[name]
        starts = []
        for r1, c1 in zip(df.orig_r1, df.orig_c1):  # best passable 500 m cell inside the 1 km origin cell
            cand = [(2 * r1 + a, 2 * c1 + b) for a in (0, 1) for b in (0, 1)]
            cand = [rc for rc in cand if np.isfinite(fr[rc])] or cand
            starts.append(min(cand, key=lambda rc: fr[rc]))
        t0 = time.time()
        with Pool(4, initializer=_init, initargs=(fr, ids, pop1, k)) as pool:
            rows = pool.map(_row, starts, chunksize=8)
        T = np.stack(rows)[:, df.uid.values - 1]  # origins x destinations (model units only)
        np.save(out, T)
        log(f"{name}: {T.shape} in {time.time() - t0:.0f}s, median {np.nanmedian(T):.2f} h, "
            f"unreachable {np.mean(~np.isfinite(T)):.4f}")


# ------------------------------------------------------------------ calibration
def load_T(name):
    T = np.load(DATA / f"T_{name}.npy").astype(float)
    T = np.where(np.isfinite(T), T, np.nan)
    T = np.nanmean(np.stack([T, T.T]), 0)  # symmetrise
    cap = 3 * np.nanmax(T)
    T = np.where(np.isfinite(T), T, cap)
    return np.maximum(T, 0.1)


def calibrate(df, T, kappa, wage_sd):
    sd_rwi = np.sqrt(np.cov(df.rwi, aweights=df["pop"]))
    w = np.exp(wage_sd / sd_rwi * (df.rwi - np.average(df.rwi, weights=df["pop"])))
    Y = (w * df["pop"]).values
    lam = (df["pop"] / df["pop"].sum()).values
    pi, _ = qse.trade_shares(Y, T ** -kappa)
    return pi, Y, lam, w.values


def built_elasticity(df):
    """Cross-sectional elasticity of built-up share w.r.t. population density (OLS, HC1 SE)."""
    d = df[(df.built_share > 0) & (df["pop"] > 0)]
    y = np.log(d.built_share.values)
    X = np.column_stack([np.ones(len(d)), np.log(d["pop"] / d.area_km2)])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi * len(y) / (len(y) - 2)
    return float(b[1]), float(np.sqrt(V[1, 1]))


def upazila_D(df):
    """Upazila means of D = log TT_pre - log TT_post to Dhaka, and module-4 panel cell counts."""
    p = pd.read_parquet(ROOT / "data/causal/panel.parquet", columns=["cell", "GID_3", "tt_pre", "tt_post", "year",
                                                                       "water_share"])
    p = p[(p.year == 2022) & (p.tt_pre > 0) & (p.tt_post > 0) & (p.water_share < 0.5)]
    p["D"] = np.log(p.tt_pre) - np.log(p.tt_post)
    g = p.groupby("GID_3").agg(D=("D", "mean"), n_cells=("cell", "size"))
    return df.join(g, on="GID_3")


def implied_rf(df, L_hat, eta):
    """Model-implied module-4 coefficient: built-up pp change regressed on D across Padma-region upazilas."""
    d = df.assign(dbuilt=100 * df.built_share * (L_hat ** eta - 1))
    d = d[d.n_cells.notna() & (d.n_cells >= 5)]
    X = np.column_stack([np.ones(len(d)), d.D.values])
    W = d.n_cells.values
    b = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * d.dbuilt.values))
    return float(b[1])


def run(df, Tpre, Tpost, sigma, alpha, alpha_N, kappa, eps, wage_sd, eta):
    pi, Y, lam, w = calibrate(df, Tpre, kappa, wage_sd)
    phi_hat = (Tpost / Tpre) ** -kappa
    sr = qse.counterfactual(pi, Y, lam, phi_hat, sigma, alpha, eps=None, alpha_N=alpha_N)
    lr = qse.counterfactual(pi, Y, lam, phi_hat, sigma, alpha, eps=eps, alpha_N=alpha_N)
    return pi, Y, lam, sr, lr, implied_rf(df, lr["L_hat"], eta)


def summarise(df, lam, sr, lr, rf):
    tr = (df.D_nat > 0.1).values
    dh = (df.NAME_2 == "Dhaka").values
    pct = lambda a: 100 * (a - 1)  # noqa: E731
    return dict(
        welfare_sr_pct=pct(sr["welfare"]), welfare_lr_pct=pct(lr["welfare"]),
        real_income_sr_treated_pct=pct(np.average(sr["V_hat"][tr], weights=lam[tr])),
        real_income_sr_rest_pct=pct(np.average(sr["V_hat"][~tr], weights=lam[~tr])),
        pop_lr_treated_pct=pct((lam[tr] * lr["L_hat"][tr]).sum() / lam[tr].sum()),
        pop_lr_dhaka_district_pct=pct((lam[dh] * lr["L_hat"][dh]).sum() / lam[dh].sum()),
        pop_lr_max_upazila_pct=pct(lr["L_hat"].max()), implied_rf_coef=rf,
        converged=bool(sr["converged"] and lr["converged"]))


def nat_D(df, g1):
    """Population-weighted D for every upazila from the national 500 m Dhaka travel-time rasters."""
    import rasterio

    ids = np.load(DATA / "ids_1km.npy")
    pop1 = np.load(DATA / "pop_1km.npy")
    H, W = ids.shape
    v = {}
    for s in ("pre", "post"):
        with rasterio.open(ROOT / f"data/travel/tt_{s}.tif") as src:
            a = src.read(1)[: 2 * H, : 2 * W].reshape(H, 2, W, 2)
        with np.errstate(all="ignore"):
            v[s] = np.nanmean(a, axis=(1, 3))
    with np.errstate(all="ignore"):
        D = np.log(v["pre"]) - np.log(v["post"])
    m = (ids > 0) & np.isfinite(D) & (pop1 > 0)
    k = int(ids.max())
    num = np.bincount(ids[m], weights=pop1[m] * D[m], minlength=k + 1)[1:]
    den = np.bincount(ids[m], weights=pop1[m], minlength=k + 1)[1:]
    with np.errstate(all="ignore"):
        return pd.Series(num / den, index=np.arange(1, k + 1))[df.uid.values].fillna(0).values


def stage_model(n_draws=600, seed=0):
    df, ids, g1 = units()
    df = upazila_D(df)
    df["D_nat"] = nat_D(df, g1)
    Tpre, Tpost = load_T("pre"), load_T("post")
    eta, eta_se = built_elasticity(df)
    log(f"units {len(df)}, built-up elasticity eta = {eta:.3f} ({eta_se:.3f})")

    # ---- main parameters
    pi, Y, lam, sr, lr, rf = run(df, Tpre, Tpost, eta=eta, **MAIN)
    main = summarise(df, lam, sr, lr, rf)
    log("main: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in main.items()}))
    own = np.diag(pi)
    calib = dict(n_units=len(df), params=MAIN, eta=eta, eta_se=eta_se, rf_coef=RF_COEF, rf_se=RF_SE,
                 own_trade_share_median=float(np.median(own)), own_trade_share_p10_p90=np.percentile(own, [10, 90]).tolist(),
                 local_goods_spending_share_median=float(np.median(MAIN['alpha_N'] / MAIN['alpha'] + (1 - MAIN['alpha_N'] / MAIN['alpha']) * own)),
                 median_tt_pre_h=float(np.median(Tpre)), share_pairs_faster_10pct=float(np.mean(Tpost < 0.9 * Tpre)),
                 treated_units_D_gt_0_1=int((df.D_nat > 0.1).sum()), treated_pop_share=float(lam[df.D_nat > 0.1].sum()),
                 main=main)
    (TAB / "qse_calibration.json").write_text(json.dumps(calib, indent=2))
    res = df[["GID_3", "NAME_1", "NAME_2", "NAME_3", "pop", "rwi", "built_share", "D_nat"]].assign(
        w_hat_sr=sr["w_hat"], V_hat_sr=sr["V_hat"], w_hat_lr=lr["w_hat"], L_hat_lr=lr["L_hat"], V_hat_lr=lr["V_hat"])
    res.to_csv(TAB / "qse_padma_main.csv", index=False)

    # ---- one-at-a-time sensitivity (incl. ferry speed)
    rows = []
    grid = dict(alpha_N=[0.0, 0.15, 0.45], kappa=[0.5, 1.5], eps=[1.5, 5.0], sigma=[4.5, 7.0], wage_sd=[0.15, 0.35])
    for k, vals in grid.items():
        for v in vals:
            p = {**MAIN, k: v}
            *_, sr_, lr_, rf_ = run(df, Tpre, Tpost, eta=eta, **p)
            rows.append(dict(varied=k, value=v, **summarise(df, lam, sr_, lr_, rf_)))
    for tag in ("slow", "fast"):
        *_, sr_, lr_, rf_ = run(df, load_T(f"pre_{tag}"), load_T(f"post_{tag}"), eta=eta, **MAIN)
        rows.append(dict(varied="ferry", value={"slow": 1.5, "fast": 4.0}[tag], **summarise(df, lam, sr_, lr_, rf_)))
    sens = pd.DataFrame([dict(varied="main", value=np.nan, **main)] + rows)
    sens.to_csv(TAB / "qse_padma_sensitivity.csv", index=False)
    log("\n" + sens.round(3).to_string())

    # ---- Bayesian layer: priors over parameters, likelihood from the module-4 estimate
    rng = np.random.default_rng(seed)
    draws = []
    t0 = time.time()
    for i in range(n_draws):
        th = dict(sigma=rng.uniform(4.5, 7.0), alpha=0.75, alpha_N=rng.uniform(0.15, 0.45), kappa=rng.uniform(0.5, 1.5),
                  eps=rng.uniform(1.0, 5.0),
                  wage_sd=rng.uniform(0.15, 0.35))
        e = rng.normal(eta, eta_se)
        phi_adj = rng.uniform(0, 1)  # share of the long-run adjustment completed by 2026
        *_, sr_, lr_, rf_ = run(df, Tpre, Tpost, eta=e, **th)
        s = summarise(df, lam, sr_, lr_, rf_)
        draws.append(dict(**th, eta=e, adj_share=phi_adj, **s,
                          loglik=-0.5 * ((RF_COEF - phi_adj * rf_) / RF_SE) ** 2))
        if (i + 1) % 100 == 0:
            log(f"draw {i + 1}/{n_draws} ({time.time() - t0:.0f}s)")
    B = pd.DataFrame(draws)
    B.to_parquet(DATA / "bayes_draws.parquet")
    wts = np.exp(B.loglik - B.loglik.max())
    wts /= wts.sum()
    keys = ["welfare_sr_pct", "welfare_lr_pct", "pop_lr_treated_pct", "implied_rf_coef", "eps", "kappa", "alpha_N", "adj_share"]

    def wq(x, w, q):
        o = np.argsort(x)
        c = np.cumsum(w[o])
        return float(np.interp(q, c / c[-1], x[o]))

    summ = {k: dict(prior_mean=float(B[k].mean()), prior_p05=float(B[k].quantile(0.05)), prior_p95=float(B[k].quantile(0.95)),
                    post_mean=float(np.sum(wts * B[k])), post_p05=wq(B[k].values, wts.values, 0.05),
                    post_p95=wq(B[k].values, wts.values, 0.95)) for k in keys}
    summ["effective_sample_size"] = float(1 / np.sum(wts ** 2))
    summ["share_draws_converged"] = float(B.converged.mean())
    summ["p_implied_rf_full_adjustment_above_rf_ci_upper"] = float(np.mean(B.implied_rf_coef > RF_COEF + 1.96 * RF_SE))
    (TAB / "qse_padma_bayes.json").write_text(json.dumps(summ, indent=2))
    log(json.dumps(summ, indent=1))
    figure(df, res, B, wts)


def figure(df, res, B, wts):
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    up = gpd.read_file(ROOT / "data/raw/gadm/gadm41_BGD_3.json")[["GID_3", "geometry"]].merge(res, on="GID_3")
    fig, ax = plt.subplots(2, 2, figsize=(13, 12))
    up.assign(v=100 * (up.V_hat_sr - 1)).plot("v", ax=ax[0, 0], cmap="viridis", legend=True,
                                               legend_kwds=dict(label="%", shrink=0.7))
    ax[0, 0].set_title("Short run (no migration): real income change, %")
    lim = np.nanpercentile(np.abs(100 * (up.L_hat_lr - 1)), 99)
    up.assign(v=100 * (up.L_hat_lr - 1)).plot("v", ax=ax[0, 1], cmap="RdBu_r", vmin=-lim, vmax=lim, legend=True,
                                               legend_kwds=dict(label="%", shrink=0.7))
    ax[0, 1].set_title("Long run (migration, eps=3): population change, %")
    for a in ax[0]:
        a.set_axis_off()
    ax[1, 0].hist(B.welfare_lr_pct, bins=40, density=True, alpha=0.5, label="prior predictive")
    ax[1, 0].hist(B.welfare_lr_pct, bins=40, weights=wts, density=True, alpha=0.5, label="posterior (given module 4)")
    ax[1, 0].set_xlabel("aggregate welfare gain, long run (%)")
    ax[1, 0].legend()
    ax[1, 1].scatter(B.implied_rf_coef, B.adj_share * B.implied_rf_coef, s=6, c=wts, cmap="viridis")
    ax[1, 1].axhspan(RF_COEF - 1.96 * RF_SE, RF_COEF + 1.96 * RF_SE, color="grey", alpha=0.25, label="module-4 95% CI")
    ax[1, 1].axhline(RF_COEF, color="k", lw=0.8)
    ax[1, 1].set_xlabel("model-implied coefficient at full long-run adjustment (pp per log point)")
    ax[1, 1].set_ylabel("implied coefficient by 2026 (x adjustment share)")
    ax[1, 1].legend()
    ax[1, 1].set_title("Can the model's long-run response be seen in 4 years of built-up data?")
    fig.tight_layout()
    fig.savefig(FIG / "qse_padma.png", dpi=110)


if __name__ == "__main__":
    {"times": stage_times, "model": stage_model}[sys.argv[1]]()
