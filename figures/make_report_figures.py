"""Result figures of the README, the technical report and the atlas, in the style of Figure 1.

Headline numbers come from the committed result tables; boundaries from GADM 4.1 level 3 (gadm41_BGD_3.json,
from gadm.org, not stored here). Four panels draw on pipeline rasters in the git-ignored data folder
(GEOAI_DATA, default ./data): the imagery-predicted wealth map (data/wealth/predictions.parquet), the 500 m
travel-time rasters (data/travel), and the city built-up maps and radial profiles (data/builtup,
data/composites). Without them the script falls back to upazila-level maps and to the pipeline's own
rendering of the city maps (outputs/figures/urban_change_raw.png).

Usage: GEOAI_DATA=path/to/data python figures/make_report_figures.py [gadm41_BGD_3.json] [figure ...]
"""
import json
import os
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
from make_figure1 import (AXIS, BLUE, BRIDGE, GADM3, GREEN, GREY, INCOME, INK, INK2, PLACES, RED, ROOT, SEA, TB,  # noqa: E402
                          detach, leader, rows, style)

FIG = ROOT / "outputs" / "figures"
DATA = Path(os.environ.get("GEOAI_DATA", ROOT / "data"))
LAND = "#f3f1ec"
C_OLD, C_NEW, C_BG = "#33415c", "#E8590C", "#f4f2ec"  # built-up before, added, background
ATLAS = ROOT / "docs" / "atlas"
SKY = "#56B4E9"
TRACK = "#d9dde1"


def seq(name, stops):
    return LinearSegmentedColormap.from_list(name, stops)


WEALTH = seq("wealth", ["#f6f1e4", "#bfe3d0", "#5ab4ac", "#2b83ba", "#0b2a66"])
BUILT = seq("built", ["#f7f3ea", "#fbd9a5", "#f29e4c", "#c8501f", "#6b1d0f"])
DENS = seq("dens", ["#f5f2f7", "#d4c6e4", "#a487c6", "#6f4aa3", "#34155e"])
ACCESS = seq("access", ["#f6f1e4", "#c7e3ec", "#74b9dc", "#2b7fbd", "#0b2a66"])
SOUTHWEST = (88.5, 91.3, 21.2, 24.2)
SOUTHWEST_TALL = (88.5, 91.3, 21.1, 24.56)
NATION = (87.9, 92.8, 20.42, 26.75)


def panel(fig, ax, s, title, dx=0.0, dy=0.012):
    """Bold panel letter and a regular-weight title above the top-left corner of the axes."""
    b = ax.get_position()
    fig.text(b.x0 + dx, b.y1 + dy, s, fontsize=8, fontweight="bold", va="baseline")
    fig.text(b.x0 + dx + 0.016, b.y1 + dy, title, fontsize=7.5, va="baseline")


class Maps:
    def __init__(self):
        U = {r["gid_3"]: r for r in rows(ATLAS / "upazila_indicators.csv")}
        g = gpd.read_file(GADM3)[["GID_3", "geometry"]]
        g = g[g.GID_3.isin(U)].copy()
        for k in next(iter(U.values())):
            if k not in ("gid_3", "division", "district", "upazila"):
                g[k] = [float(U[i][k]) for i in g.GID_3]
        g["division"] = [U[i]["division"] for i in g.GID_3]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            merged = g.geometry.buffer(0.004).union_all().buffer(-0.004)
        self.g = g
        self.coast = gpd.GeoSeries([q.exterior for q in getattr(merged, "geoms", [merged]) if q.area > 0.002])

    def base(self, ax, lw=0.0):
        ax.set_facecolor(SEA)
        self.g.plot(ax=ax, color=LAND, edgecolor="white", linewidth=lw, zorder=1)

    def finish(self, ax, extent=NATION, places=(), bridge=False, fs=6.5):
        self.coast.plot(ax=ax, color=AXIS, linewidth=0.4, zorder=4)
        halo = [pe.withStroke(linewidth=1.6, foreground="white")]
        for name in places:
            x, y = PLACES[name]
            ax.plot(x, y, marker="o", ms=2.4, mfc="white", mec=INK, mew=0.7, zorder=5)
            left = name in ("Chattogram", "Sylhet")
            ax.text(x + (-0.06 if left else 0.06), y + 0.02, name, fontsize=fs, ha="right" if left else "left",
                    va="center", path_effects=halo, zorder=5)
        if bridge:
            ax.plot(*BRIDGE, marker="s", ms=3.2, mfc=INK, mec="white", mew=0.7, zorder=6)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect(1 / 0.916)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)

    def draw(self, ax, col, cmap, norm, extent=NATION, places=(), bridge=False, lw=0.12, fs=6.5):
        ax.set_facecolor(SEA)
        self.g.plot(column=col, cmap=cmap, norm=norm, ax=ax, edgecolor="white", linewidth=lw, zorder=1)
        self.finish(ax, extent, places, bridge, fs)

    def bar(self, fig, ax, cmap, norm, label, ticks, box=(0.05, 0.09, 0.42, 0.017), extend="neither", fmt=None):
        cax = ax.inset_axes(box)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                          ticks=ticks, extend=extend)
        cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.5, labelsize=6.5, pad=1.5)
        cb.ax.minorticks_off()
        if fmt:
            cb.ax.set_xticklabels(fmt)
        halo = [pe.withStroke(linewidth=1.8, foreground="white")]
        cb.ax.set_title(label, fontsize=7, loc="left", pad=3).set_path_effects(halo)
        for t in cb.ax.get_xticklabels():
            t.set_path_effects(halo)


def dumbbell(ax, y, a, b, ca=GREY, cb=BLUE, open_a=True):
    ax.plot([a, b], [y, y], color=TRACK, lw=2.6, solid_capstyle="round", zorder=1)
    ax.plot(a, y, marker="o", ms=4.4, mfc="white" if open_a else ca, mec=ca, mew=1.0, zorder=2)
    ax.plot(b, y, marker="o", ms=5, color=cb, zorder=3)


def rowaxis(ax):
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_position(("outward", 3))


def tt_reduction(extent, step=0.004):
    """Percentage cut in least-cost travel time to Dhaka (500 m rasters), resampled to a lon/lat grid."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    with rasterio.open(DATA / "travel" / "tt_pre.tif") as src:
        pre, tr, crs = src.read(1).astype("float32"), src.transform, src.crs
    with rasterio.open(DATA / "travel" / "tt_post.tif") as src:
        post = src.read(1).astype("float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        red = np.where(pre > 0, 100 * (1 - post / pre), np.nan).astype("float32")
    x0, x1, y0, y1 = extent
    w, h = int(round((x1 - x0) / step)), int(round((y1 - y0) / step))
    dtr = from_origin(x0, y1, step, step)
    out = np.full((h, w), np.nan, "float32")
    reproject(red, out, src_transform=tr, src_crs=crs, dst_transform=dtr, dst_crs="EPSG:4326",
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return out, dtr


def land_mask(M, shape, transform):
    from rasterio.features import geometry_mask

    return geometry_mask(M.g.geometry, shape, transform, invert=True)


def city_rasters():
    """Built-up masks (2020, 2026) and radial profiles from the pipeline rasters, using the urban-expansion code."""
    import runpy

    if not (DATA / "builtup").exists():
        return None
    m = runpy.run_path(str(ROOT / "scripts" / "03_urban_expansion.py"), run_name="figures")
    g = m["load_city"].__globals__  # run_path returns a copy; patch the functions' own globals
    g["PRED"], g["COMP"] = DATA / "builtup", DATA / "composites"
    Y, MAIN = m["YEARS"], m["MAIN"]
    out = {}
    for name, c in json.loads((ROOT / "configs" / "aois.json").read_text())["cities"].items():
        P, CL, tr, res = m["load_city"](name)
        idx = [Y.index(y) for y in MAIN]
        valid = np.isfinite(P[idx]).all(0) & (CL[idx] > 0).all(0)
        S = m["temporal_smooth"](P, Y)
        b0 = (S[Y.index(m["T0"])] >= 0.5) & valid
        b1 = ((S[Y.index(m["T1"])] >= 0.5) & valid) | b0
        r, s0 = m["radial_profile"](b0, tr, res, c["lon"], c["lat"], valid, max_km=c["half_km"])
        _, s1 = m["radial_profile"](b1, tr, res, c["lon"], c["lat"], valid, max_km=c["half_km"])
        out[name] = dict(b0=b0, b1=b1, valid=valid, r=r, s0=s0, s1=s1, res=res)
    return out


# ------------------------------------------------------------------ atlas: six national maps
def atlas(M):
    fig = plt.figure(figsize=(7.2, 6.75), dpi=300)
    inc = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=4)
    pop = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=8)
    spec = [("rwi_meta", WEALTH, Normalize(-0.4, 1.2), "Relative Wealth Index", [-0.4, 0, 0.4, 0.8, 1.2], "max", None,
             "Wealth (Meta index, 2021)"),
            ("built_share_wc2021_pct", BUILT, LogNorm(0.1, 100), "Built-up share of area (%)", [0.1, 1, 10, 100],
             "min", ["0.1", "1", "10", "100"], "Built-up land (WorldCover 2021)"),
            ("pop_density_per_km2", DENS, LogNorm(100, 100000), "People per km²", [100, 1000, 10000, 100000], "both",
             ["100", "1k", "10k", "100k"], "Population density (2022)"),
            ("tt_reduction_pct", ACCESS, Normalize(0, 70), "Travel time to Dhaka cut (%)", [0, 35, 70], "neither",
             None, "Access gain from the bridge"),
            ("real_income_sr_pct", INCOME, inc, "Short-run real income change (%)", [-1, 0, 2, 4], "max", None,
             "Model: real income, no migration"),
            ("population_lr_pct", INCOME, pop, "Long-run population change (%)", [-2, 0, 4, 8], "max", None,
             "Model: population, with migration")]
    W, H = 0.318, 0.455
    for i, (col, cmap, norm, label, ticks, ext, fmt, title) in enumerate(spec):
        ax = fig.add_axes([0.008 + (i % 3) * 0.332, 0.505 - (i // 3) * 0.495, W, H])
        M.draw(ax, col, cmap, norm, places=("Dhaka",) if i < 3 else (), bridge=i >= 3, lw=0.08)
        M.bar(fig, ax, cmap, norm, label, ticks, box=(0.05, 0.085, 0.50, 0.02), extend=ext, fmt=fmt)
        panel(fig, ax, "abcdef"[i], title, dy=0.006)
    fig.savefig(ATLAS / "atlas_maps.png", dpi=300)


# ------------------------------------------------------------------ atlas: southwest zoom
def southwest(M):
    fig = plt.figure(figsize=(7.2, 2.85), dpi=300)
    spec = [("tt_reduction_pct", ACCESS, Normalize(0, 70), "Travel time to Dhaka cut (%)", [0, 35, 70], "neither",
             "Access gain"),
            ("real_income_sr_pct", INCOME, TwoSlopeNorm(vmin=-1, vcenter=0, vmax=8), "Short-run real income (%)",
             [-1, 0, 4, 8], "max", "Real income, no migration"),
            ("population_lr_pct", INCOME, TwoSlopeNorm(vmin=-2, vcenter=0, vmax=14), "Long-run population (%)",
             [-2, 0, 7, 14], "max", "Population, with migration")]
    for i, (col, cmap, norm, label, ticks, ext, title) in enumerate(spec):
        ax = fig.add_axes([0.008 + i * 0.332, 0.02, 0.318, 0.88])
        M.draw(ax, col, cmap, norm, extent=SOUTHWEST, places=("Dhaka", "Khulna", "Barishal"), bridge=True, lw=0.25)
        M.bar(fig, ax, cmap, norm, label, ticks, box=(0.05, 0.085, 0.46, 0.022), extend=ext)
        if i == 0:
            ax.annotate("Padma Bridge", xy=BRIDGE, xytext=(89.0, 23.95), fontsize=6.5, va="center",
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, shrinkB=2.5, mutation_scale=6))
        panel(fig, ax, "abc"[i], title, dy=0.012)
    fig.savefig(ATLAS / "atlas_maps_southwest.png", dpi=300)


# ------------------------------------------------------------------ Padma Bridge: exposure, event study, robustness
def padma(M):
    fig = plt.figure(figsize=(7.2, 5.0), dpi=300)
    S = json.loads((TB / "padma_sample_description.json").read_text())
    a = fig.add_axes([0.012, 0.03, 0.465, 0.905])
    norm = Normalize(0, 70)
    raster = (DATA / "travel" / "tt_pre.tif").exists()
    if raster:
        M.base(a)
        red, dtr = tt_reduction(SOUTHWEST_TALL)
        red[~land_mask(M, red.shape, dtr)] = np.nan
        x0, x1, y0, y1 = SOUTHWEST_TALL
        a.imshow(red, extent=(x0, x0 + red.shape[1] * dtr.a, y1 + red.shape[0] * dtr.e, y1), cmap=ACCESS, norm=norm,
                 interpolation="nearest", zorder=2)
        M.finish(a, SOUTHWEST_TALL, places=("Dhaka", "Khulna", "Barishal"), bridge=True)
    else:
        M.draw(a, "tt_reduction_pct", ACCESS, norm, extent=SOUTHWEST_TALL, places=("Dhaka", "Khulna", "Barishal"),
               bridge=True, lw=0.3)
    a.annotate("Padma Bridge", xy=BRIDGE, xytext=(89.05, 24.05), fontsize=6.5, va="center",
               arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, shrinkB=2.5, mutation_scale=6))
    M.bar(fig, a, ACCESS, norm, "Modeled travel time to Dhaka cut (%)", [0, 10, 30, 50, 70], box=(0.05, 0.09, 0.46, 0.016))
    a.text(0.05, 0.022, "500 m grid, least-cost routes with and without the bridge" if raster else
           f'{S["n_cells"]:,} cells in {S["n_upazila"]} upazilas; map shows upazila means',
           transform=a.transAxes, fontsize=6.5, color=INK2)
    fig.text(0.012, 0.957, "a", fontsize=8, fontweight="bold", va="baseline")
    fig.text(0.028, 0.957, "Exposure: reduction in travel time to Dhaka due to the bridge", fontsize=7.5, va="baseline")

    # b: event study, three specifications, 2018-2026 sample
    b = fig.add_axes([0.60, 0.70, 0.385, 0.235])
    E = rows(TB / "padma_event_study.csv")
    b.axvspan(2017.7, 2019.5, color="#f0f1f2", lw=0)
    b.text(2017.8, 4.5, "2018–19: older archive,\nnot comparable", fontsize=6.5, color=INK2, va="top", linespacing=1.15)
    b.axhline(0, color=AXIS, lw=0.5)
    b.axvline(2022.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
    b.text(2022.58, 4.2, "bridge opens", fontsize=6.5, color=INK2, va="center")
    P = {}
    for spec, col in (("(1)", RED), ("(2)", BLUE), ("(3)", GREEN)):
        get = lambda smp: sorted((int(r["year"]), float(r["coef"]), float(r["ci_lo"]), float(r["ci_hi"])) for r in E
                                 if r["spec"].startswith(spec) and r["sample"].startswith(smp))
        old = [p for p in get("2018") if p[0] <= 2020]
        pts = sorted(get("2020") + [(2022, 0.0, 0.0, 0.0)])
        b.plot([p[0] for p in old], [p[1] for p in old], color=col, lw=0.8, ls=(0, (2, 2)), marker="o", ms=2.2,
               mfc="white", mew=0.7, alpha=0.75)
        x = [p[0] for p in pts]
        b.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.14, lw=0)
        b.plot(x, [p[1] for p in pts], color=col, lw=1.1, marker="o", ms=2.8, mfc="white", mew=0.9)
        P[spec] = dict((p[0], p[1]) for p in pts)
    leader(b, (2025, P["(1)"][2025]), (2022.75, -1.75), "(1) cell and year FE", ha="left")
    leader(b, (2025, P["(2)"][2025]), (2022.75, 2.6), "(2) + baseline-density trends", ha="left")
    leader(b, (2021, P["(3)"][2021]), (2019.75, 1.5), "(3) + division × year", ha="left")
    b.set_xlim(2017.7, 2026.2); b.set_ylim(-2.2, 4.7); b.set_xticks(range(2018, 2027, 2)); b.set_yticks([-2, 0, 2])
    b.set_ylabel("Built-up share\n(pp per log point)")
    detach(b); panel(fig, b, "b", "Pre-trends are absent once density trends are controlled", dx=-0.103)

    # c: specification (2) under alternative choices
    c = fig.add_axes([0.72, 0.085, 0.265, 0.48])
    R = {r["model"]: r for r in rows(TB / "padma_did_results.csv")}
    order = [("main: (1) cell + year FE", "(1) Cell and year FE", RED),
             ("main: (2) + year x baseline-built decile, year x dist/water", "(2) + density trends", BLUE),
             ("main: (3) + division x year FE", "(3) + division × year", GREEN),
             ("cluster by district", "(2), district clusters", BLUE),
             ("ferry 1.5 km/h (bigger gains)", "Ferry 1.5 km/h", GREY), ("ferry 4 km/h (smaller gains)", "Ferry 4 km/h", GREY),
             ("binary treatment D > 0.2", "Binary treatment", GREY),
             ("outcome: mean built-up probability", "Outcome: probability", GREY),
             ("drop cells < 10 km from bridge", "Drop < 10 km of bridge", GREY),
             ("drop cells < 30 km from Dhaka", "Drop < 30 km of Dhaka", GREY),
             ("heterogeneity: baseline built < 5%", "Baseline built < 5%", GREY)]
    c.axvline(0, color=AXIS, lw=0.5)
    c.axhspan(len(order) - 3.5, len(order) - 0.5, color="#f3f4f5", lw=0, zorder=0)
    for i, (k, name, col) in enumerate(order):
        y = len(order) - 1 - i
        r = R[k]
        c.plot([float(r["ci_lo"]), float(r["ci_hi"])], [y, y], color=col, lw=1.1, solid_capstyle="round")
        c.plot(float(r["coef"]), y, marker="o", ms=4, mfc=col if i < 4 else "white", mec=col, mew=1.0)
    c.set_yticks(range(len(order)), [o[1] for o in order][::-1])
    c.set_xlim(-0.62, 0.30); c.set_ylim(-0.6, len(order) - 0.4); c.set_xticks([-0.6, -0.3, 0, 0.3])
    c.set_xlabel("Effect on built-up share (pp per log point)", labelpad=2, x=0.2)
    rowaxis(c); panel(fig, c, "c", "The null is robust to alternative specifications (95% CI)", dx=-0.223)
    fig.savefig(FIG / "padma_treatment_and_event_study.png", dpi=300)


# ------------------------------------------------------------------ step 4: incidence, sensitivity, detectability
def model(M):
    fig = plt.figure(figsize=(7.2, 5.0), dpi=300)
    C = json.loads((TB / "qse_calibration.json").read_text())
    B = json.loads((TB / "qse_padma_bayes.json").read_text())
    Pw = json.loads((TB / "padma_power.json").read_text())
    a = fig.add_axes([0.012, 0.03, 0.465, 0.905])
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=8)
    M.draw(a, "population_lr_pct", INCOME, norm, places=PLACES, bridge=True, lw=0.15)
    a.annotate("Padma\nBridge", xy=BRIDGE, xytext=(87.98, 23.3), fontsize=6.5, va="center", linespacing=1.15,
               arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, shrinkB=2.5, mutation_scale=6))
    M.bar(fig, a, INCOME, norm, "Long-run population change (%)", [-2, 0, 2, 4, 6, 8], extend="max")
    a.text(0.05, 0.022, f'Treated upazilas +{C["main"]["pop_lr_treated_pct"]:.1f}%; largest '
           f'+{C["main"]["pop_lr_max_upazila_pct"]:.0f}%', transform=a.transAxes, fontsize=6.5, color=INK2)
    fig.text(0.012, 0.957, "a", fontsize=8, fontweight="bold", va="baseline")
    fig.text(0.028, 0.957, "Model-implied long-run population change", fontsize=7.5, va="baseline")

    # inset: who gains, by wealth fifth of the population (short-run real income, model)
    I = json.loads((TB / "qse_incidence.json").read_text())
    ins = a.inset_axes([0.585, 0.835, 0.37, 0.125])
    ins.set_facecolor("none")
    fifths = [f["real_income_sr_pct"] for f in I["wealth_fifths"]]
    ins.bar(range(1, 6), fifths, width=0.68, color=[BLUE if v == max(fifths) else SKY for v in fifths])
    for k, v in enumerate(fifths, 1):
        ins.text(k, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ins.set_xticks([1, 5], ["poorest fifth", "richest"]); ins.set_yticks([0, 0.5]); ins.set_ylim(0, 0.78)
    ins.tick_params(labelsize=6, length=1.8, pad=1.5)
    ins.set_title("Income gain by wealth fifth (%)", fontsize=6.5, loc="left", pad=3)
    detach(ins)

    # b: welfare gain under alternative parameter values
    b = fig.add_axes([0.685, 0.665, 0.30, 0.27])
    S = rows(TB / "qse_padma_sensitivity.csv")
    main = float(S[0]["welfare_sr_pct"])
    labels = {"alpha_N": "Non-traded share", "ferry": "Ferry speed (km/h)", "kappa": "Trade-cost elasticity",
              "sigma": "Substitution elasticity", "wage_sd": "Wage dispersion", "eps": "Migration elasticity"}
    groups = {}
    for r in S[1:]:
        groups.setdefault(r["varied"], []).append((float(r["value"]), float(r["welfare_lr_pct" if r["varied"] == "eps"
                                                                              else "welfare_sr_pct"])))
    order = sorted(groups, key=lambda k: max(v for _, v in groups[k]) - min(v for _, v in groups[k]))
    b.axvline(main, color=INK2, lw=0.6, ls=(0, (3, 3)))
    for i, k in enumerate(order):
        pts = sorted(groups[k], key=lambda t: t[1])
        b.plot([pts[0][1], pts[-1][1]], [i, i], color=TRACK, lw=2.6, solid_capstyle="round", zorder=1)
        for val, w in pts:
            b.plot(w, i, marker="o", ms=4.4, color=BLUE, zorder=3)
        if pts[-1][1] - pts[0][1] > 0.05:
            b.text(pts[0][1] - 0.022, i, f"{pts[0][0]:g}", ha="right", va="center", fontsize=6.3, color=INK2)
            b.text(pts[-1][1] + 0.022, i, f"{pts[-1][0]:g}", ha="left", va="center", fontsize=6.3, color=INK2)
    b.text(main + 0.012, len(order) - 0.42, f"main {main:.2f}%", fontsize=6.5, color=INK2, va="bottom")
    b.set_yticks(range(len(order)), [labels[k] for k in order])
    b.set_xlim(0.2, 0.9); b.set_ylim(-0.55, len(order) - 0.1); b.set_xticks([0.2, 0.4, 0.6, 0.8])
    b.set_xlabel("National welfare gain (%)", labelpad=2)
    rowaxis(b); panel(fig, b, "b", "Welfare gain 0.30–0.79% across parameter values", dx=-0.188)

    # c: what the model implies, what was estimated, and the smallest effect the design could detect
    c = fig.add_axes([0.685, 0.40, 0.30, 0.125])
    est, se = C["rf_coef"], Pw["standard_errors"]
    mde, ratio = Pw["minimum_detectable_effect_80pct_power"], Pw["mde_relative_to_model_effect"]
    q = B["implied_rf_coef"]
    c.axvline(0, color=AXIS, lw=0.5)
    c.plot([q["post_p05"], q["post_p95"]], [2, 2], color=RED, lw=2.4, solid_capstyle="round")
    c.plot(C["main"]["implied_rf_coef"], 2, marker="o", ms=4.6, mfc="white", mec=RED, mew=1.1)
    for y, k in ((1, "upazila clusters"), (0, "district clusters")):
        c.plot([est - 1.96 * se[k], est + 1.96 * se[k]], [y, y], color=BLUE, lw=1.1, solid_capstyle="round")
        c.plot(est, y, marker="o", ms=4.4, color=BLUE)
        c.plot([mde[k]] * 2, [y - 0.3, y + 0.3], color=INK, lw=1.3, solid_capstyle="butt")
        c.text(max(mde[k], est + 1.96 * se[k]) + 0.012, y, f"{ratio[k]:.1f}× model", va="center", fontsize=6.5)
    c.set_yticks([2, 1, 0], ["Model implies", "Estimate, upazila clusters", "Estimate, district clusters"])
    c.set_xlim(-0.14, 0.37); c.set_ylim(-0.55, 2.5); c.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    c.set_xlabel("Built-up share (pp per log point)", labelpad=2)
    rowaxis(c); panel(fig, c, "c", "Minimum detectable effect (bar): 2–4 times the model-implied effect", dx=-0.188)

    # d: the model-implied effect in a typical treated cell, in pixels (schematic, with an enlarged window)
    tc = Pw["typical_treated_cell"]
    n = int(round(tc["pixels_per_cell"] ** 0.5))
    hamlet = [(19, 17), (19, 18), (20, 18), (20, 19), (21, 19), (21, 20)][: int(round(tc["baseline_built_pixels"]))]
    side = tc["model_implied_effect_pixels"] ** 0.5
    grid = np.zeros((n, n))
    for r_, c_ in hamlet:
        grid[r_, c_] = 1
    cell_cmap = LinearSegmentedColormap.from_list("cell", [C_BG, C_OLD])
    w0, z0, z1 = 0.135, 15, 25  # panel width (figure fraction); enlarged window in pixel coordinates
    d = fig.add_axes([0.535, 0.04, w0, w0 * 7.2 / 5.0])
    zo = fig.add_axes([0.705, 0.04, w0, w0 * 7.2 / 5.0])
    for ax, lo, hi, step, lw in ((d, 0, n, 5, 0.3), (zo, z0, z1, 1, 0.5)):
        ax.imshow(grid, cmap=cell_cmap, vmin=0, vmax=1, extent=(0, n, n, 0), interpolation="nearest")
        for g_ in range(lo, hi + 1, step):
            ax.axhline(g_, color="white", lw=lw); ax.axvline(g_, color="white", lw=lw)
        ax.add_patch(plt.Rectangle((21, 21), side, side, color=C_NEW, lw=0, zorder=3))
        ax.set_xlim(lo, hi); ax.set_ylim(hi, lo); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#aeb4ba"); sp.set_linewidth(0.6)
    d.add_patch(plt.Rectangle((z0, z0), z1 - z0, z1 - z0, fill=False, ec=INK, lw=0.6, zorder=4))
    from matplotlib.patches import ConnectionPatch
    for ya in (z0, z1):  # zoom guides from the window in the full cell to the enlarged view
        fig.add_artist(ConnectionPatch(xyA=(z1, ya), coordsA=d.transData, xyB=(z0, ya), coordsB=zo.transData,
                                       color=INK2, lw=0.4))
    d.text(n / 2, n + 1.8, f'1.2 km cell, {tc["pixels_per_cell"]:,} pixels', fontsize=6, ha="center", va="top",
           color=INK2)
    zo.text((z0 + z1) / 2, z1 + 0.45, "enlarged 4× (schematic)", fontsize=6, ha="center", va="top", color=INK2)
    panel(fig, d, "d", "Model-implied effect in a typical treated cell: 0.4 of one pixel", dx=-0.038)
    tx, ty = 0.872, zo.get_position().y1 - 0.03
    blocks = [(C_OLD, "Built-up before", f'{tc["baseline_built_pp"]:.2f}% of the cell,', f'about {tc["baseline_built_pixels"]:.0f} pixels'),
              (C_NEW, "Model implies", f'+{tc["model_implied_effect_pp"]:.3f} pp, about', f'{tc["model_implied_effect_pixels"]:.1f} of one pixel')]
    for k, (col, head, l1, l2) in enumerate(blocks):
        y0 = ty - k * 0.092
        fig.add_artist(plt.Rectangle((tx - 0.014, y0 - 0.001), 0.0075, 0.0075 * 7.2 / 5.0, transform=fig.transFigure,
                                     color=col, lw=0))
        fig.text(tx, y0, head, fontsize=6.5, va="baseline", fontweight="bold")
        fig.text(tx, y0 - 0.024, l1, fontsize=6.5, va="baseline")
        fig.text(tx, y0 - 0.048, l2, fontsize=6.5, va="baseline")
    fig.savefig(FIG / "qse_padma.png", dpi=300)


# ------------------------------------------------------------------ step 1: measurement
def measurement(M):
    import pandas as pd

    fig = plt.figure(figsize=(7.2, 5.0), dpi=300)
    a = fig.add_axes([0.012, 0.03, 0.465, 0.905])
    norm = Normalize(-0.6, 1.2)
    P = DATA / "wealth" / "predictions.parquet"
    pts = pd.read_parquet(P) if P.exists() else None
    if pts is not None:
        M.base(a)
        a.scatter(pts.longitude, pts.latitude, c=pts.rwi_pred_imagery_context, cmap=WEALTH, norm=norm, s=1.15,
                  marker="s", lw=0, zorder=2, rasterized=True)
        M.finish(a, places=PLACES)
        title, note = ("Wealth predicted from imagery, out of area",
                       f"{len(pts):,} grid cells; out-of-fold predictions, 50 km spatial blocks")
    else:
        M.draw(a, "rwi_meta", WEALTH, norm, places=PLACES, lw=0.15)
        title, note = "The target: relative wealth across 544 upazilas", "Label: Meta Relative Wealth Index"
    M.bar(fig, a, WEALTH, norm, "Relative Wealth Index" + (", predicted" if pts is not None else ""),
          [-0.6, 0, 0.6, 1.2], extend="both")
    a.text(0.05, 0.022, note, transform=a.transAxes, fontsize=6.5, color=INK2)
    fig.text(0.012, 0.957, "a", fontsize=8, fontweight="bold", va="baseline")
    fig.text(0.028, 0.957, title, fontsize=7.5, va="baseline")

    Wr = rows(TB / "wealth_prediction_results.csv")
    names = {"context only: pop+travel time (gbm)": "Pop. + travel time", "s2_stats (gbm)": "Sentinel-2 statistics",
             "mosaiks (ridge)": "Random conv. features", "landcover (gbm)": "Land cover",
             "s2_stats+landcover (gbm)": "S2 + land cover", "imagery all: s2+lc+mosaiks50 (gbm)": "All imagery",
             "imagery + context (gbm)": "Imagery + context"}
    get = lambda cv: {r["features"]: float(r["r2_upazila"]) for r in Wr if r["target"] == "rwi" and r["cv"].startswith(cv)}  # noqa: E731
    sp, rd = get("spatial"), get("random")
    order = sorted(names, key=lambda k: sp[k])

    # b: upazila means, prediction against label
    b = fig.add_axes([0.665, 0.70, 0.185, 0.235])
    if pts is not None:
        u = pts.groupby("GID_3")[["rwi", "rwi_pred_imagery_context"]].mean()
        b.plot([-0.6, 1.6], [-0.6, 1.6], color=INK2, lw=0.6, ls=(0, (3, 3)))
        b.scatter(u.rwi, u.rwi_pred_imagery_context, s=4, color=BLUE, lw=0.2, edgecolor="white", alpha=0.85)
        b.set_xlim(-0.6, 1.6); b.set_ylim(-0.6, 1.6); b.set_xticks([0, 1]); b.set_yticks([0, 1])
        b.set_aspect("equal", adjustable="box")
        b.set_xlabel("Label, upazila mean", labelpad=2); b.set_ylabel("Predicted", labelpad=2)
        b.text(1.55, -0.45, f"{len(u)} upazilas", ha="right", fontsize=6.5, color=INK2)
        detach(b); panel(fig, b, "b", "Predicted and labeled upazila means agree", dx=-0.06)
    else:
        b.set_visible(False)

    # c: accuracy by feature set, spatial-block against random folds
    c = fig.add_axes([0.72, 0.33, 0.265, 0.215])
    for i, k in enumerate(order):
        dumbbell(c, i, rd[k], sp[k])
        c.text(sp[k] - 0.016, i, f"{sp[k]:.2f}", ha="right", va="center", fontsize=6.3,
               fontweight="bold" if i == len(order) - 1 else "normal")
    leader(c, (sp[order[1]], 1), (0.37, 2.55), "spatial-block CV", ha="left")
    leader(c, (rd[order[0]], 0), (0.56, 0.0), "random CV", ha="left")
    c.set_yticks(range(len(order)), [names[k] for k in order]); c.tick_params(axis="y", labelsize=6.3)
    c.set_xlim(0.36, 0.95); c.set_ylim(-0.6, len(order) - 0.4); c.set_xticks([0.4, 0.6, 0.8])
    c.set_xlabel("Out-of-sample R², upazila wealth", labelpad=2)
    rowaxis(c); panel(fig, c, "c", "Random folds overstate out-of-area accuracy", dx=-0.223)

    # d: year-to-year noise of built-up maps under the two compositing designs
    d = fig.add_axes([0.72, 0.085, 0.265, 0.135])
    U = sorted(rows(TB / "urban_expansion_summary.csv"), key=lambda r: float(r["reversal_rate_v1_least_cloudy"]))
    for i, r in enumerate(U):
        dumbbell(d, i, 100 * float(r["reversal_rate_v1_least_cloudy"]), 100 * float(r["reversal_rate"]))
    top = U[-1]
    leader(d, (100 * float(top["reversal_rate_v1_least_cloudy"]), len(U) - 1), (17.5, len(U) - 0.2), "least-cloudy",
           ha="left")
    leader(d, (100 * float(top["reversal_rate"]), len(U) - 1), (4.0, len(U) - 0.2), "balanced", ha="left")
    d.set_yticks(range(len(U)), [r["city"].title() for r in U]); d.tick_params(axis="y", labelsize=6.3)
    d.set_xlim(0, 27); d.set_ylim(-0.6, len(U) + 0.1); d.set_xticks([0, 10, 20])
    d.set_xlabel("Built pixels that flip back next year (%)", labelpad=2)
    rowaxis(d); panel(fig, d, "d", "Phenology-balanced compositing reduces noise", dx=-0.223)
    fig.savefig(FIG / "measurement.png", dpi=300)


# ------------------------------------------------------------------ step 2: six city maps


def city_maps(R=None):
    from PIL import Image

    cfg = json.loads((ROOT / "configs" / "aois.json").read_text())["cities"]
    U = {r["city"]: r for r in rows(TB / "urban_expansion_summary.csv")}
    rgb = {k: np.array(matplotlib.colors.to_rgb(v)) for k, v in (("old", C_OLD), ("new", C_NEW), ("bg", C_BG),
                                                                  ("sea", SEA))}
    imgs = {}
    if R:
        for name, d in R.items():
            img = np.ones(d["b0"].shape + (3,)) * rgb["bg"]
            img[~d["valid"]] = rgb["sea"]
            img[d["b0"]] = rgb["old"]
            img[d["b1"] & ~d["b0"]] = rgb["new"]
            imgs[name] = (img, d["b0"].shape[0])
    else:  # recolor the pipeline's own 110 dpi rendering, unmixing the two classes per pixel
        raw = np.asarray(Image.open(FIG / "urban_change_raw.png").convert("RGB")).astype(float) / 255
        old, new, nodata = np.array([0.35] * 3), np.array([0.85, 0.15, 0.1]), np.array([0.8, 0.85, 0.95])
        A = np.stack([1 - old, 1 - new], axis=1)
        sol = np.clip((1 - raw.reshape(-1, 3)) @ np.linalg.pinv(A).T, 0, 1)
        sol = sol / np.maximum(sol.sum(1, keepdims=True), 1)
        img = (rgb["bg"] * (1 - sol.sum(1, keepdims=True)) + sol[:, :1] * rgb["old"] + sol[:, 1:] * rgb["new"]).reshape(raw.shape)
        img[np.abs(raw - nodata).sum(2) < 0.06] = rgb["sea"]
        for i, name in enumerate(cfg):
            x0, y0 = 25 + 544 * (i % 3), 33 if i < 3 else 573
            imgs[name] = (img[y0:y0 + 512, x0:x0 + 512], 512)
    fig = plt.figure(figsize=(7.2, 5.15), dpi=300)
    halo = [pe.withStroke(linewidth=1.6, foreground=C_BG)]
    for i, name in enumerate(cfg):
        img, n = imgs[name]
        ax = fig.add_axes([0.008 + (i % 3) * 0.332, 0.505 - (i // 3) * 0.485, 0.318, 0.318 * 7.2 / 5.15])
        ax.imshow(img, interpolation="antialiased", extent=(0, n, n, 0))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c9ced3"); sp.set_linewidth(0.5)
        r, side = U[name], 2 * cfg[name]["half_km"]
        px = n * 10 / side
        ax.plot([0.953 * n - px, 0.953 * n], [0.965 * n] * 2, color=INK, lw=1.0, solid_capstyle="butt")
        ax.text(0.953 * n - px / 2, 0.95 * n, "10 km", ha="center", va="bottom", fontsize=6, path_effects=halo)
        bx = ax.get_position()
        fig.text(bx.x0, bx.y1 + 0.008, "abcdef"[i], fontsize=8, fontweight="bold", va="baseline")
        fig.text(bx.x0 + 0.016, bx.y1 + 0.008, f'{name.title()}  +{float(r["growth_pct"]):.0f}%', fontsize=7.5,
                 va="baseline")
        fig.text(bx.x1, bx.y1 + 0.008, f'{float(r["built_km2_t0"]):.0f} → {float(r["built_km2_t1"]):.0f} km²',
                 fontsize=6.5, va="baseline", ha="right", color=INK2)
        if i == 2:
            for j, (col, txt) in enumerate(((C_OLD, "built-up in 2020"), (C_NEW, "added 2020–2026"))):
                ax.plot(0.05 * n, (0.05 + 0.058 * j) * n, marker="s", ms=4, color=col, mec="white", mew=0.4)
                ax.text(0.082 * n, (0.05 + 0.058 * j) * n, txt, fontsize=6.5, va="center", path_effects=halo)
        ax.set_xlim(0, n); ax.set_ylim(n, 0)
    fig.savefig(FIG / "urban_change_maps.png", dpi=300)


# ------------------------------------------------------------------ step 2: series, where growth happens, gradients
def urban_series(R=None):
    fig = plt.figure(figsize=(7.2, 2.75), dpi=300)
    A = rows(TB / "urban_built_area_by_year.csv")
    U = rows(TB / "urban_expansion_summary.csv")
    cities = [r["city"] for r in sorted(U, key=lambda r: -float(r["growth_pct"]))]
    hi = {cities[0]: BLUE, cities[-1]: INK}
    a = fig.add_axes([0.065, 0.165, 0.235, 0.69])
    ends = {}
    for name in cities:
        s = sorted((int(r["year"]), float(r["built_km2_smoothed"])) for r in A if r["city"] == name and r["built_km2_smoothed"])
        col = hi.get(name, "#9aa7b4")
        a.plot([p[0] for p in s], [100 * p[1] / s[0][1] for p in s], color=col, lw=1.4 if name in hi else 0.9,
               zorder=3 if name in hi else 2)
        ends[name] = (100 * s[-1][1] / s[0][1], col)
    ys = sorted(ends.items(), key=lambda kv: kv[1][0])
    pos = [v[0] for _, v in ys]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + 5.4)
    for (name, (y, col)), p in zip(ys, pos):
        a.plot([2026.05, 2026.3], [y, p], color=AXIS, lw=0.4, clip_on=False)
        a.text(2026.36, p, name.title(), fontsize=6.5, va="center", fontweight="bold" if col == BLUE else "normal")
    a.set_xlim(2020, 2026); a.set_ylim(98, 176); a.set_xticks([2020, 2023, 2026]); a.set_yticks([100, 125, 150, 175])
    a.set_ylabel("Built-up area (2020 = 100)")
    detach(a); panel(fig, a, "a", "Built-up area, smoothed", dx=-0.055, dy=0.03)

    b = fig.add_axes([0.45, 0.165, 0.25, 0.69])
    if R:
        for name in cities:
            d = R[name]
            col = {"dhaka": INK, "chattogram": GREEN}.get(name, "#b8c2cc")
            b.plot(d["r"], 100 * (d["s1"] - d["s0"]), color=col, lw=1.3 if col != "#b8c2cc" else 0.8,
                   zorder=3 if col != "#b8c2cc" else 2)
        pk = {k: (R[k]["r"][np.nanargmax(R[k]["s1"] - R[k]["s0"])], 100 * np.nanmax(R[k]["s1"] - R[k]["s0"]))
              for k in ("dhaka", "chattogram")}
        leader(b, pk["chattogram"], (8.0, 15.5), "Chattogram", ha="left")
        leader(b, pk["dhaka"], (11.0, 11.8), "Dhaka", ha="left")
        leader(b, (1.2, 15.0), (3.2, 19.0), "four smaller cities", ha="left", color=INK2)
        b.set_xlim(0, 25); b.set_ylim(0, 21); b.set_xticks([0, 5, 10, 15, 20, 25]); b.set_yticks([0, 10, 20])
        b.set_xlabel("Distance from the city center (km)", labelpad=2)
        b.set_ylabel("Built-up share added\n2020–2026 (pp)")
        detach(b); panel(fig, b, "b", "Two largest cities: additions peak at 4–7 km", dx=-0.06, dy=0.03)
    else:
        b.set_visible(False)

    c = fig.add_axes([0.84, 0.165, 0.145, 0.69])
    G = sorted([r for r in U if r["gradient_b_t0"]], key=lambda r: float(r["gradient_b_t0"]))
    for i, r in enumerate(G):
        dumbbell(c, i, float(r["gradient_b_t0"]), float(r["gradient_b_t1"]))
    leader(c, (float(G[-1]["gradient_b_t0"]), len(G) - 1), (0.62, len(G) - 0.35), "2020", ha="left")
    leader(c, (float(G[-1]["gradient_b_t1"]), len(G) - 1), (0.30, len(G) - 0.35), "2026", ha="left")
    c.set_yticks(range(len(G)), [r["city"].title() for r in G])
    c.set_xlim(0, 1.0); c.set_ylim(-0.55, len(G) - 0.1); c.set_xticks([0, 0.5, 1.0])
    c.set_xlabel("Density gradient (per km)", labelpad=2)
    rowaxis(c); panel(fig, c, "c", "Density gradients flatten", dx=-0.085, dy=0.03)
    fig.savefig(FIG / "urban_area_and_profiles.png", dpi=300)


if __name__ == "__main__":
    style()
    which = sys.argv[2:] or ["atlas", "southwest", "padma", "model", "measurement", "city_maps", "urban_series"]
    M = Maps() if set(which) & {"atlas", "southwest", "padma", "model", "measurement"} else None
    R = city_rasters() if set(which) & {"city_maps", "urban_series"} else None
    for name in which:
        f = globals()[name]
        f(M) if name in ("atlas", "southwest", "padma", "model", "measurement") else f(R)
        plt.close("all")
