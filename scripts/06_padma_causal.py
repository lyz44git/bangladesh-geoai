"""Did the Padma Bridge (opened 25 June 2022) accelerate built-up growth in the areas it connected?

Design: continuous-treatment difference-in-differences / event study on a
balanced panel of 1.2 km cells, 2020-2026 dry seasons (Collection 1 imagery).
Treatment intensity D_i = log(travel time to Dhaka without the bridge)
                         - log(travel time with the bridge),
from the least-cost travel-time model (scripts/04_travel_time.py). D = 0 for
places whose best route to Dhaka does not use the bridge.

  Y_it = a_i + l_t + sum_{k != 2022} b_k * D_i * 1[t = k] + controls_t + e_it

The 2022 composite (Nov 2021 - Mar 2022) is the last pre-opening season.
Standard errors are clustered by upazila.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoai import labels  # noqa: E402
from geoai.builtup import grid_of, read_prob  # noqa: E402
from geoai.econ import feols  # noqa: E402
from geoai.grid import Grid, lonlat_to_tm, tm_to_lonlat  # noqa: E402

CFG = json.loads((ROOT / "configs/aois.json").read_text())
PRED, TRAVEL = ROOT / "data/builtup", ROOT / "data/travel"
TAB, FIG, DATA = ROOT / "outputs/tables", ROOT / "outputs/figures", ROOT / "data/causal"
DATA.mkdir(parents=True, exist_ok=True)
CELL = 40  # 40 x 30 m = 1.2 km
YEARS = CFG["years"]
MAIN = [y for y in YEARS if y >= 2020]
REF = 2022
DHAKA = (90.40, 23.78)
BRIDGE = (90.262, 23.446)  # centroid of the bridge segments found in Overture (see padma_bridge_segments.csv)


def build_panel():
    out = DATA / "panel.parquet"
    if out.exists():
        return pd.read_parquet(out)
    g = grid_of(str(PRED / f"padma_{REF}.tif"))
    H, W = g.height // CELL, g.width // CELL
    cg = Grid(g.xmin, g.ymax, W, H, g.res * CELL)
    recs = {}
    for y in YEARS:
        p = read_prob(PRED / f"padma_{y}.tif")[: H * CELL, : W * CELL]
        valid = np.isfinite(p)
        b = (p >= 0.5) & valid
        blk = lambda a: a.reshape(H, CELL, W, CELL).sum((1, 3))  # noqa: E731
        nv = blk(valid.astype(np.float32))
        recs[(y, "valid")] = nv / CELL**2
        with np.errstate(invalid="ignore", divide="ignore"):
            recs[(y, "built")] = blk(b.astype(np.float32)) / nv
            recs[(y, "prob")] = blk(np.nan_to_num(p).astype(np.float32)) / nv
    rr, cc = np.mgrid[0:H, 0:W]
    x = cg.xmin + (cc + 0.5) * cg.res
    yy = cg.ymax - (rr + 0.5) * cg.res
    lon, lat = tm_to_lonlat(x.ravel(), yy.ravel())
    base = pd.DataFrame({"cell": np.arange(H * W), "x": x.ravel(), "y": yy.ravel(), "lon": lon, "lat": lat})
    base["water_share"] = labels.class_fraction(cg, labels.WATER, year=2021).ravel()
    for name in ("pre", "post", "pre_slow", "post_slow", "pre_fast", "post_fast"):
        with rasterio.open(TRAVEL / f"tt_{name}.tif") as src:
            base[f"tt_{name}"] = [v[0] for v in src.sample(zip(base.x, base.y))]
    up = gpd.read_file(ROOT / "data/raw/gadm/gadm41_BGD_3.json")[["NAME_1", "GID_2", "NAME_2", "GID_3", "NAME_3", "geometry"]]
    pts = gpd.GeoDataFrame(base[["cell"]], geometry=gpd.points_from_xy(base.lon, base.lat), crs=4326)
    j = gpd.sjoin(pts, up, how="left", predicate="within").drop_duplicates("cell").set_index("cell")
    base = base.join(j[["NAME_1", "GID_2", "NAME_2", "GID_3", "NAME_3"]], on="cell")
    dx, dy = lonlat_to_tm(*DHAKA)
    bx, by = lonlat_to_tm(*BRIDGE)
    base["dist_dhaka_km"] = np.hypot(base.x - dx, base.y - dy) / 1000
    base["dist_bridge_km"] = np.hypot(base.x - bx, base.y - by) / 1000
    rows = []
    for y in YEARS:
        d = base.copy()
        d["year"] = y
        d["valid"] = recs[(y, "valid")].ravel()
        d["built"] = 100 * recs[(y, "built")].ravel()  # percent of cell area
        d["prob"] = 100 * recs[(y, "prob")].ravel()
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(out)
    return panel


def prepare(panel, years):
    p = panel[panel.year.isin(years) & panel.GID_3.notna()].copy()
    # Destination cells (travel time 0) have no defined log change in access.
    p = p[(p.water_share < 0.5) & np.isfinite(p.tt_pre) & np.isfinite(p.tt_post) & (p.tt_post > 0) & (p.tt_pre > 0)]
    ok = p.groupby("cell").valid.min() >= 0.8
    p = p[p.cell.isin(ok[ok].index)]
    n_years = p.groupby("cell").year.nunique()
    p = p[p.cell.isin(n_years[n_years == len(years)].index)].copy()
    p["D"] = np.log(p.tt_pre) - np.log(p.tt_post)
    p["D_slow"] = np.log(p.tt_pre_slow) - np.log(p.tt_post_slow)
    p["D_fast"] = np.log(p.tt_pre_fast) - np.log(p.tt_post_fast)
    p["post"] = (p.year > REF).astype(float)
    pre = p[p.year <= REF].groupby("cell").built.mean().rename("built_pre")
    p = p.join(pre, on="cell")
    # Baseline-built bins are defined per cell (not per observation): bin 0 = no
    # built-up before opening, bins 1-9 = quantiles of the positive values.
    cb = pre.copy()
    bins = pd.Series(0, index=cb.index)
    pos = cb > 0
    bins[pos] = 1 + pd.qcut(cb[pos], 9, labels=False, duplicates="drop")
    p = p.join(bins.rename("built_pre_bin"), on="cell")
    p["log_dist_dhaka"] = np.log(p.dist_dhaka_km)
    return p


def add_event_terms(p, years, var="D"):
    names = []
    for y in years:
        if y == REF:
            continue
        n = f"{var}_x_{y}"
        p[n] = p[var] * (p.year == y)
        names.append(n)
    return names


def add_year_controls(p, years):
    """Year-specific slopes on log distance to Dhaka and on 2021 water share (river dynamics)."""
    names = []
    for y in years[1:]:
        for v in ("log_dist_dhaka", "water_share"):
            n = f"{v}_x_{y}"
            p[n] = p[v] * (p.year == y)
            names.append(n)
    return names


SPECS = {
    "(1) cell + year FE": dict(fe=["cell", "year"], ctrl=False),
    "(2) + year x baseline-built decile, year x dist/water": dict(fe=["cell", "year^built_pre_bin"], ctrl=True),
    "(3) + division x year FE": dict(fe=["cell", "year^built_pre_bin", "year^NAME_1"], ctrl=True),
}


def event_study(p, years, label):
    rows = []
    for sname, s in SPECS.items():
        q = p.copy()
        ev = add_event_terms(q, years)
        ctrl = add_year_controls(q, years) if s["ctrl"] else []
        res, _ = feols(q, "built", ev + ctrl, s["fe"], "GID_3")
        r = res.loc[ev].copy()
        r["year"] = [int(n.split("_")[-1]) for n in ev]
        r["spec"], r["sample"] = sname, label
        rows.append(r)
    return pd.concat(rows)


def did(p, var="D", outcome="built", spec="(2) + year x baseline-built decile, year x dist/water", years=MAIN,
        cluster="GID_3"):
    q = p.copy()
    q["treat_post"] = q[var] * q.post
    s = SPECS[spec]
    ctrl = add_year_controls(q, years) if s["ctrl"] else []
    res, _ = feols(q, outcome, ["treat_post"] + ctrl, s["fe"], cluster)
    return res.loc["treat_post"]


def randomization_inference(p, n_perm=500, seed=0):
    """Permute upazila-level mean treatment across upazilas within division (spec 1, closed form)."""
    rng = np.random.default_rng(seed)
    q = p.copy()
    yd = q.built - q.groupby("cell").built.transform("mean") - q.groupby("year").built.transform("mean") + q.built.mean()
    post_c = q.post - q.post.mean()
    up_D = q.groupby("GID_3").D.mean()
    up_div = q.groupby("GID_3").NAME_1.first()

    def coef(Dcell):
        x = (Dcell - Dcell.mean()) * post_c  # two-way demeaned D_i x Post_t in a balanced panel
        return float((x * yd).sum() / (x * x).sum())

    obs = coef(q.GID_3.map(up_D).values)
    null = []
    for _ in range(n_perm):
        perm = up_D.copy()
        for div, idx in up_div.groupby(up_div).groups.items():
            perm.loc[idx] = rng.permutation(up_D.loc[idx].values)
        null.append(coef(q.GID_3.map(perm).values))
    null = np.array(null)
    return dict(coef_upazila_mean_D=obs, p_two_sided=float(np.mean(np.abs(null) >= abs(obs))), n_perm=n_perm)


def randomization_inference_fwl(p, spec, n_perm=1000, seed=0):
    """Randomization inference for the pooled DiD of specs (1)-(3), closed form.

    In a balanced panel whose controls all have the form z_i x year, Frisch-Waugh
    reduces the partialled-out regressor to D~_i (post_t - mean post), where D~ is
    the cross-sectional residual of D on z (baseline-built decile dummies, log
    distance to Dhaka, water share, and for spec (3) division dummies). So
    coef = sum_i D~_i Delta_i / (sum_i D~_i^2 * sum_t (post_t - mean post)^2) with
    Delta_i = sum_t (post_t - mean post) y_it. Upazila-mean D is permuted across
    upazilas within division.
    """
    rng = np.random.default_rng(seed)
    q = p.sort_values(["cell", "year"])
    pc = q.post - q.post.mean()
    delta = (q.built * pc).groupby(q.cell).sum()
    c = q.drop_duplicates("cell").set_index("cell").loc[delta.index]
    pc2 = float((pc ** 2).sum() / len(delta))
    Z = [np.ones(len(c))]
    if spec >= 2:
        Z += [(c.built_pre_bin == b).astype(float).values for b in sorted(c.built_pre_bin.unique())[1:]]
        Z += [c.log_dist_dhaka.values, c.water_share.values]
    if spec >= 3:
        Z += [(c.NAME_1 == d).astype(float).values for d in sorted(c.NAME_1.unique())[1:]]
    Qz, _ = np.linalg.qr(np.column_stack(Z))
    dv = delta.values

    def coef(Dc):
        Dt = Dc - Qz @ (Qz.T @ Dc)
        return float(Dt @ dv / ((Dt ** 2).sum() * pc2))

    up_D = c.groupby("GID_3").D.mean()
    up_div = c.groupby("GID_3").NAME_1.first()
    obs = coef(c.GID_3.map(up_D).values)
    null = []
    for _ in range(n_perm):
        perm = up_D.copy()
        for _, idx in up_div.groupby(up_div).groups.items():
            perm.loc[idx] = rng.permutation(up_D.loc[idx].values)
        null.append(coef(c.GID_3.map(perm).values))
    null = np.array(null)
    return dict(spec=spec, coef_cell_D=coef(c.D.values), coef_upazila_mean_D=obs,
                p_two_sided=float(np.mean(np.abs(null) >= abs(obs))), n_perm=n_perm)


def run_ri(p):
    out = [randomization_inference_fwl(p, s) for s in (1, 2, 3)]
    (TAB / "padma_randomization_inference.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(out).round(4).to_string())


def main():
    if sys.argv[1:] == ["ri"]:
        return run_ri(prepare(build_panel(), MAIN))
    panel = build_panel()
    p = prepare(panel, MAIN)
    cells = p.drop_duplicates("cell")
    desc = dict(n_cells=int(len(cells)), n_obs=int(len(p)), n_upazila=int(cells.GID_3.nunique()),
                n_district=int(cells.GID_2.nunique()), share_treated_D_gt_0_05=float((cells.D > 0.05).mean()),
                D_mean_treated=float(cells.D[cells.D > 0.05].mean()), D_p50_treated=float(cells.D[cells.D > 0.05].median()),
                D_p90=float(cells.D.quantile(0.9)), built_pre_mean=float(cells.built_pre.mean()),
                built_pre_mean_treated=float(cells.built_pre[cells.D > 0.05].mean()),
                built_pre_mean_control=float(cells.built_pre[cells.D <= 0.05].mean()))
    (TAB / "padma_sample_description.json").write_text(json.dumps(desc, indent=2))
    print(json.dumps(desc, indent=2))

    es = event_study(p, MAIN, "2020-2026 (Collection 1)")
    p_ext = prepare(panel, YEARS)
    es_ext = event_study(p_ext, YEARS, "2018-2026 (incl. older L2A)")
    es_all = pd.concat([es, es_ext])
    es_all.to_csv(TAB / "padma_event_study.csv")

    # Pooled DiD and robustness
    rows = []

    def add(name, r, n):
        rows.append(dict(model=name, coef=r.coef, se=r.se, ci_lo=r.ci_lo, ci_hi=r.ci_hi, n_obs=n))

    for sname in SPECS:
        add(f"main: {sname}", did(p, spec=sname), len(p))
    add("cluster by district", did(p, cluster="GID_2"), len(p))
    add("ferry 1.5 km/h (bigger gains)", did(p, var="D_slow"), len(p))
    add("ferry 4 km/h (smaller gains)", did(p, var="D_fast"), len(p))
    q = p.copy()
    q["Dbin"] = (q.D > 0.2).astype(float)
    add("binary treatment D > 0.2", did(q, var="Dbin"), len(q))
    add("outcome: mean built-up probability", did(p, outcome="prob"), len(p))
    far = p[p.dist_bridge_km > 10]
    add("drop cells < 10 km from bridge", did(far), len(far))
    nodhaka = p[p.dist_dhaka_km > 30]
    add("drop cells < 30 km from Dhaka", did(nodhaka), len(nodhaka))
    towns = p[p.built_pre >= 5]
    rural = p[p.built_pre < 5]
    add("heterogeneity: baseline built >= 5%", did(towns), len(towns))
    add("heterogeneity: baseline built < 5%", did(rural), len(rural))
    # placebo: pretend the bridge opened in 2020 using only pre-opening seasons 2018-2022
    pp = prepare(panel, [2018, 2019, 2020, 2021, 2022])
    pp["post"] = (pp.year >= 2020).astype(float)
    add("placebo: fake opening 2020 (2018-2022 only)",
        did(pp, spec="(1) cell + year FE", years=[2018, 2019, 2020, 2021, 2022]), len(pp))
    dd = pd.DataFrame(rows)
    dd["effect_per_0.5_logpt"] = dd.coef * 0.5
    dd.to_csv(TAB / "padma_did_results.csv", index=False)
    print(dd.round(4).to_string())
    run_ri(p)

    # Figures: treatment map and event study
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    sc = ax[0].scatter(cells.lon, cells.lat, c=cells.D, s=1.2, cmap="viridis", vmin=0, vmax=1.0)
    ax[0].plot(*BRIDGE, marker="*", color="red", ms=14)
    ax[0].plot(*DHAKA, marker="s", color="black", ms=7)
    plt.colorbar(sc, ax=ax[0], label="D = log travel-time reduction to Dhaka")
    ax[0].set_title("Treatment intensity (red star: bridge, black square: Dhaka)")
    for (sname, sample), g in es_all.groupby(["spec", "sample"]):
        if sample.startswith("2018") and not sname.startswith("(2)"):
            continue
        g = pd.concat([g, pd.DataFrame({"year": [REF], "coef": [0.0], "ci_lo": [0.0], "ci_hi": [0.0]})]).sort_values("year")
        off = {"(1)": -0.15, "(2)": 0.0, "(3)": 0.15}[sname[:3]] + (0.05 if sample.startswith("2018") else 0)
        lbl = sname if sample.startswith("2020") else f"{sname[:3]} extended 2018-"
        ax[1].errorbar(g.year + off, 0.5 * g.coef, yerr=[0.5 * (g.coef - g.ci_lo), 0.5 * (g.ci_hi - g.coef)],
                       marker="o", capsize=2, label=lbl, ls="--" if sample.startswith("2018") else "-")
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].axvline(REF + 0.5, color="red", lw=0.8, ls=":")
    ax[1].set_xlabel("dry season ending in year (bridge opened June 2022)")
    ax[1].set_ylabel("effect on built-up share (pp) of a 0.5 log-pt TT cut")
    ax[1].set_title("Event study, 95% CI clustered by upazila")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "padma_treatment_and_event_study.png", dpi=110)


if __name__ == "__main__":
    main()
