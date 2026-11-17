"""Result figures of the README, the technical report and the atlas, in the style of Figure 1.

Drawn from the committed result tables and the GADM 4.1 level-3 boundaries (gadm41_BGD_3.json, from
gadm.org, not stored in this repository). The city change maps are recoloured from the pipeline's
own rendering of the classified rasters (outputs/figures/urban_change_raw.png, written by
scripts/03_urban_expansion.py), because the rasters themselves are not stored here.

Usage: python figures/make_report_figures.py [path/to/gadm41_BGD_3.json]
"""
import json
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
    fig.text(b.x0 + dx, b.y1 + dy, s, fontsize=10, fontweight="bold", va="bottom")
    fig.text(b.x0 + dx + 0.022, b.y1 + dy + 0.0015 * 5 / fig.get_figheight(), title, fontsize=7.5, va="bottom")


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

    def draw(self, ax, col, cmap, norm, extent=NATION, places=(), bridge=False, lw=0.12, fs=6.5):
        ax.set_facecolor(SEA)
        self.g.plot(column=col, cmap=cmap, norm=norm, ax=ax, edgecolor="white", linewidth=lw)
        self.coast.plot(ax=ax, color=AXIS, linewidth=0.4)
        halo = [pe.withStroke(linewidth=1.6, foreground="white")]
        for name in places:
            x, y = PLACES[name]
            ax.plot(x, y, marker="o", ms=2.4, mfc="white", mec=INK, mew=0.7)
            left = name in ("Chattogram", "Sylhet")
            ax.text(x + (-0.06 if left else 0.06), y + 0.02, name, fontsize=fs, ha="right" if left else "left",
                    va="center", path_effects=halo)
        if bridge:
            ax.plot(*BRIDGE, marker="s", ms=3.2, mfc=INK, mec="white", mew=0.7, zorder=5)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect(1 / 0.916)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

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


# ------------------------------------------------------------------ atlas: south-west zoom
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
    M.draw(a, "tt_reduction_pct", ACCESS, norm, extent=SOUTHWEST_TALL, places=("Dhaka", "Khulna", "Barishal"),
           bridge=True, lw=0.3)
    a.annotate("Padma Bridge", xy=BRIDGE, xytext=(89.05, 24.05), fontsize=6.5, va="center",
               arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, shrinkB=2.5, mutation_scale=6))
    M.bar(fig, a, ACCESS, norm, "Modelled travel time to Dhaka cut (%)", [0, 10, 30, 50, 70], box=(0.05, 0.09, 0.46, 0.016))
    a.text(0.05, 0.022, f'{S["n_cells"]:,} cells in {S["n_upazila"]} upazilas; map shows upazila means',
           transform=a.transAxes, fontsize=6.5, color=INK2)
    fig.text(0.012, 0.957, "a", fontsize=10, fontweight="bold", va="bottom")
    fig.text(0.034, 0.9585, "Exposure: how much the bridge shortens the trip to Dhaka", fontsize=7.5, va="bottom")

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
    detach(b); panel(fig, b, "b", "Pre-trends vanish once density trends are controlled", dx=-0.103)

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
    rowaxis(c); panel(fig, c, "c", "The null holds across alternative choices (95% CI)", dx=-0.223)
    fig.savefig(FIG / "padma_treatment_and_event_study.png", dpi=300)


# ------------------------------------------------------------------ spatial model: incidence, sensitivity, power
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
    fig.text(0.012, 0.957, "a", fontsize=10, fontweight="bold", va="bottom")
    fig.text(0.034, 0.9585, "Where people move once migration adjusts (model)", fontsize=7.5, va="bottom")

    # b: welfare gain under alternative parameter values
    b = fig.add_axes([0.685, 0.60, 0.30, 0.335])
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
            b.plot(w, i, marker="o", ms=4.6, color=BLUE, zorder=3)
        if pts[-1][1] - pts[0][1] > 0.05:
            b.text(pts[0][1] - 0.022, i, f"{pts[0][0]:g}", ha="right", va="center", fontsize=6.3, color=INK2)
            b.text(pts[-1][1] + 0.022, i, f"{pts[-1][0]:g}", ha="left", va="center", fontsize=6.3, color=INK2)
    b.text(main + 0.012, len(order) - 0.42, f"main {main:.2f}%", fontsize=6.5, color=INK2, va="bottom")
    b.set_yticks(range(len(order)), [labels[k] for k in order])
    b.set_xlim(0.2, 0.9); b.set_ylim(-0.55, len(order) - 0.1); b.set_xticks([0.2, 0.4, 0.6, 0.8])
    b.set_xlabel("National welfare gain (%)", labelpad=2)
    rowaxis(b); panel(fig, b, "b", "Welfare gain 0.30–0.79% across parameter values", dx=-0.188)

    # c: the model's implied reduced-form coefficient against the estimate
    c = fig.add_axes([0.685, 0.335, 0.30, 0.125])
    est, se_u, se_d = C["rf_coef"], Pw["standard_errors"]["upazila clusters"], Pw["standard_errors"]["district clusters"]
    q = B["implied_rf_coef"]
    c.axvline(0, color=AXIS, lw=0.5)
    c.plot([q["post_p05"], q["post_p95"]], [2, 2], color=RED, lw=2.4, solid_capstyle="round")
    c.plot(C["main"]["implied_rf_coef"], 2, marker="o", ms=4.6, mfc="white", mec=RED, mew=1.1)
    for y, se in ((1, se_u), (0, se_d)):
        c.plot([est - 1.96 * se, est + 1.96 * se], [y, y], color=BLUE, lw=1.1, solid_capstyle="round")
        c.plot(est, y, marker="o", ms=4.4, color=BLUE)
    c.set_yticks([2, 1, 0], ["Model implies", "Estimate, upazila clusters", "Estimate, district clusters"])
    c.set_xlim(-0.14, 0.26); c.set_ylim(-0.5, 2.5); c.set_xticks([-0.1, 0, 0.1, 0.2])
    c.set_xlabel("Built-up share (pp per log point)", labelpad=2)
    rowaxis(c); panel(fig, c, "c", "The model's effect sits at the point estimate", dx=-0.188)

    # d: power of the causal design against the model-implied effect
    d = fig.add_axes([0.685, 0.085, 0.30, 0.11])
    pw = Pw["power"]
    items = [("Full adjustment", "main parameters, full adjustment"), ("Half adjustment", "main parameters, half adjustment")]
    for i, (name, k) in enumerate(items[::-1]):
        u, dd = 100 * pw[k]["upazila clusters"], 100 * pw[k]["district clusters"]
        dumbbell(d, i, dd, u, ca=GREY, cb=BLUE)
        d.text(u + 2.5, i, f"{u:.0f}%" + (" upazila clusters" if i else ""), va="center", fontsize=6.5)
        d.text(dd - 2.5, i, f"{dd:.0f}%" if not i else f"district {dd:.0f}%", va="center", ha="right", fontsize=6.5,
               color=INK2)
    d.axvline(80, color=INK2, lw=0.6, ls=(0, (3, 3)))
    d.text(79, 1.5, "80% target", ha="right", va="center", fontsize=6.5, color=INK2)
    d.set_yticks([1, 0], [i[0] for i in items])
    d.set_xlim(-22, 85); d.set_ylim(-0.55, 1.9); d.set_xticks([0, 20, 40, 60, 80])
    d.set_xlabel("Power to detect the model-implied effect (%)", labelpad=2)
    rowaxis(d); panel(fig, d, "d", "The design has little power against it", dx=-0.188)
    fig.savefig(FIG / "qse_padma.png", dpi=300)


# ------------------------------------------------------------------ wealth mapping
def wealth(M):
    fig = plt.figure(figsize=(7.2, 5.0), dpi=300)
    a = fig.add_axes([0.012, 0.03, 0.465, 0.905])
    norm = Normalize(-0.4, 1.2)
    M.draw(a, "rwi_meta", WEALTH, norm, places=PLACES, lw=0.15)
    M.bar(fig, a, WEALTH, norm, "Relative Wealth Index, upazila mean", [-0.4, 0, 0.4, 0.8, 1.2], extend="max")
    a.text(0.05, 0.022, "Label: Meta Relative Wealth Index, 26,544 grid cells", transform=a.transAxes, fontsize=6.5,
           color=INK2)
    fig.text(0.012, 0.957, "a", fontsize=10, fontweight="bold", va="bottom")
    fig.text(0.034, 0.9585, "The target: relative wealth across 544 upazilas", fontsize=7.5, va="bottom")

    Wr = rows(TB / "wealth_prediction_results.csv")
    names = {"context only: pop+travel time (gbm)": "Pop. + travel time", "s2_stats (gbm)": "Sentinel-2 statistics",
             "mosaiks (ridge)": "Random conv. features", "landcover (gbm)": "Land cover",
             "s2_stats+landcover (gbm)": "S2 + land cover", "imagery all: s2+lc+mosaiks50 (gbm)": "All imagery",
             "imagery + context (gbm)": "Imagery + context"}
    get = lambda tgt, cv, col: {r["features"]: float(r[col]) for r in Wr if r["target"] == tgt and r["cv"].startswith(cv)}
    sp, rd, cell = get("rwi", "spatial", "r2_upazila"), get("rwi", "random", "r2_upazila"), get("rwi", "spatial", "r2")
    fm = get("flowminder_wipov2011", "spatial", "r2_upazila")
    order = sorted(names, key=lambda k: sp[k])

    # b: all feature sets, upazila means, spatial-block against random folds
    b = fig.add_axes([0.72, 0.56, 0.265, 0.375])
    for i, k in enumerate(order):
        dumbbell(b, i, rd[k], sp[k])
        b.text(sp[k] - 0.016, i, f"{sp[k]:.2f}", ha="right", va="center", fontsize=6.5,
               fontweight="bold" if i == len(order) - 1 else "normal")
    leader(b, (sp[order[1]], 1), (0.37, 2.0), "spatial-block CV", ha="left")
    leader(b, (rd[order[0]], 0), (0.56, 0.0), "random CV", ha="left")
    b.set_yticks(range(len(order)), [names[k] for k in order])
    b.set_xlim(0.36, 0.95); b.set_ylim(-0.55, len(order) - 0.45); b.set_xticks([0.4, 0.6, 0.8])
    b.set_xlabel("Out-of-sample R², upazila means", labelpad=2)
    rowaxis(b); panel(fig, b, "b", "Random folds overstate accuracy for every feature set", dx=-0.223)

    # c: the same models scored against a second, independent poverty map
    c = fig.add_axes([0.60, 0.085, 0.155, 0.31])
    c.plot([0.38, 0.9], [0.38, 0.9], color=INK2, lw=0.6, ls=(0, (3, 3)))
    for k in order:
        c.plot(sp[k], fm[k], marker="o", ms=4.6, color=BLUE, mec="white", mew=0.6)
    leader(c, (sp[order[0]], fm[order[0]]), (0.50, 0.40), names[order[0]], ha="left")
    leader(c, (sp[order[-1]], fm[order[-1]]), (0.40, 0.86), names[order[-1]], ha="left")
    c.set_xlim(0.38, 0.9); c.set_ylim(0.38, 0.9); c.set_xticks([0.4, 0.6, 0.8]); c.set_yticks([0.4, 0.6, 0.8])
    c.set_aspect("equal", adjustable="box")
    c.set_xlabel("R², Meta wealth index", labelpad=2); c.set_ylabel("R², 2011 poverty map", labelpad=2)
    detach(c); panel(fig, c, "c", "Same ranking on a second label", dx=-0.07)

    # d: where the predictive power comes from
    d = fig.add_axes([0.865, 0.085, 0.12, 0.31])
    imp = rows(TB / "wealth_feature_importance.csv")
    lab = {"lc_built": "Built-up", "lc_crop": "Cropland", "lc_tree": "Tree cover"}
    top = [(lab[r["feature"]], 100 * float(r["gain_share"]), BLUE) for r in imp[:3]]
    rest = {"s2": "Sentinel-2 (61)", "mosaiks": "Conv. feat. (50)", "lc": "Other land cover"}
    for g, name in rest.items():
        tot = 100 * sum(float(r["gain_share"]) for r in imp[3:] if r["feature"].startswith(g + "_"))
        top.append((name, tot, SKY if g != "lc" else SKY))
    top = [t for t in top if t[1] > 0.5][::-1]
    for i, (name, v, col) in enumerate(top):
        d.barh(i, v, height=0.62, color=col)
        d.text(v + 1.2, i, f"{v:.0f}", va="center", fontsize=6.5)
    d.set_yticks(range(len(top)), [t[0] for t in top])
    d.set_xlim(0, 40); d.set_ylim(-0.6, len(top) - 0.4); d.set_xticks([0, 20, 40])
    d.set_xlabel("Share of model gain (%)", labelpad=2)
    rowaxis(d); panel(fig, d, "d", "Built-up land leads", dx=-0.088)
    fig.savefig(FIG / "wealth_maps_and_accuracy.png", dpi=300)


# ------------------------------------------------------------------ urban expansion: six city maps
def city_maps():
    from PIL import Image
    cfg = json.loads((ROOT / "configs" / "aois.json").read_text())["cities"]
    U = {r["city"]: r for r in rows(TB / "urban_expansion_summary.csv")}
    raw = np.asarray(Image.open(FIG / "urban_change_raw.png").convert("RGB")).astype(float) / 255
    old, new, nodata = np.array([0.35] * 3), np.array([0.85, 0.15, 0.1]), np.array([0.8, 0.85, 0.95])
    C_OLD, C_NEW, C_BG = (np.array(matplotlib.colors.to_rgb(c)) for c in ("#33415c", "#E8590C", "#f4f2ec"))
    # unmix each pixel into shares of the two classes (the rendering averaged neighbouring pixels)
    A = np.stack([1 - old, 1 - new], axis=1)
    sol = np.clip((1 - raw.reshape(-1, 3)) @ np.linalg.pinv(A).T, 0, 1)
    tot = np.maximum(sol.sum(1, keepdims=True), 1)
    sol = sol / tot
    img = (C_BG * (1 - sol.sum(1, keepdims=True)) + sol[:, :1] * C_OLD + sol[:, 1:] * C_NEW).reshape(raw.shape)
    img[np.abs(raw - nodata).sum(2) < 0.06] = matplotlib.colors.to_rgb(SEA)
    fig = plt.figure(figsize=(7.2, 5.15), dpi=300)
    names = list(cfg)
    for i, name in enumerate(names):
        x0, y0 = 25 + 544 * (i % 3), 33 if i < 3 else 573
        ax = fig.add_axes([0.008 + (i % 3) * 0.332, 0.505 - (i // 3) * 0.485, 0.318, 0.318 * 7.2 / 5.15])
        ax.imshow(img[y0:y0 + 512, x0:x0 + 512], interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#c9ced3"); s.set_linewidth(0.5)
        r, side = U[name], 2 * cfg[name]["half_km"]
        px = 512 * 10 / side
        ax.plot([488 - px, 488], [494, 494], color=INK, lw=1.0, solid_capstyle="butt")
        ax.text(488 - px / 2, 486, "10 km", ha="center", va="bottom", fontsize=6, path_effects=[
            pe.withStroke(linewidth=1.4, foreground=matplotlib.colors.to_hex(C_BG))])
        b = ax.get_position()
        fig.text(b.x0, b.y1 + 0.008, "abcdef"[i], fontsize=10, fontweight="bold", va="bottom")
        fig.text(b.x0 + 0.022, b.y1 + 0.0095, f'{name.title()}  +{float(r["growth_pct"]):.0f}%', fontsize=7.5, va="bottom")
        fig.text(b.x1, b.y1 + 0.0095, f'{float(r["built_km2_t0"]):.0f} → {float(r["built_km2_t1"]):.0f} km²',
                 fontsize=6.5, va="bottom", ha="right", color=INK2)
        if i == 2:
            for j, (col, txt) in enumerate(((C_OLD, "built-up in 2020"), (C_NEW, "added 2020–2026"))):
                ax.plot(26, 26 + j * 30, marker="s", ms=4, color=col, mec="white", mew=0.4)
                ax.text(42, 26 + j * 30, txt, fontsize=6.5, va="center", path_effects=[
                    pe.withStroke(linewidth=1.6, foreground=matplotlib.colors.to_hex(C_BG))])
        ax.set_xlim(0, 511); ax.set_ylim(511, 0)
    fig.savefig(FIG / "urban_change_maps.png", dpi=300)


# ------------------------------------------------------------------ urban expansion: series, gradients, noise
def urban_series():
    fig = plt.figure(figsize=(7.2, 2.75), dpi=300)
    A = rows(TB / "urban_built_area_by_year.csv")
    U = rows(TB / "urban_expansion_summary.csv")
    cities = [r["city"] for r in sorted(U, key=lambda r: -float(r["growth_pct"]))]
    a = fig.add_axes([0.065, 0.165, 0.285, 0.69])
    ends = {}
    for name in cities:
        s = sorted((int(r["year"]), float(r["built_km2_smoothed"]), float(r["built_km2_raw"])) for r in A
                   if r["city"] == name and r["built_km2_smoothed"])
        base = s[0][1]
        hi = name in (cities[0], cities[-1])
        col = BLUE if name == cities[0] else (INK if name == cities[-1] else "#9aa7b4")
        a.plot([p[0] for p in s], [100 * p[1] / base for p in s], color=col, lw=1.4 if hi else 0.9, zorder=3 if hi else 2)
        ends[name] = (100 * s[-1][1] / base, col)
    ys = sorted(ends.items(), key=lambda kv: kv[1][0])
    pos = [v[0] for _, v in ys]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + 5.2)
    for (name, (y, col)), p in zip(ys, pos):
        a.plot([2026.05, 2026.3], [y, p], color=AXIS, lw=0.4, clip_on=False)
        a.text(2026.36, p, name.title(), fontsize=6.5, va="center",
               color=INK, fontweight="bold" if col == BLUE else "normal")
    a.set_xlim(2020, 2026); a.set_ylim(98, 176); a.set_xticks([2020, 2022, 2024, 2026]); a.set_yticks([100, 125, 150, 175])
    a.set_ylabel("Built-up area (2020 = 100)")
    detach(a); panel(fig, a, "a", "Built-up area, smoothed annual series", dx=-0.055, dy=0.03)

    b = fig.add_axes([0.575, 0.165, 0.155, 0.69])
    G = sorted([r for r in U if r["gradient_b_t0"]], key=lambda r: float(r["gradient_b_t0"]))
    for i, r in enumerate(G):
        dumbbell(b, i, float(r["gradient_b_t0"]), float(r["gradient_b_t1"]))
    leader(b, (float(G[-1]["gradient_b_t0"]), len(G) - 1), (0.62, len(G) - 0.35), "2020", ha="left")
    leader(b, (float(G[-1]["gradient_b_t1"]), len(G) - 1), (0.36, len(G) - 0.35), "2026", ha="left")
    b.set_yticks(range(len(G)), [r["city"].title() for r in G])
    b.set_xlim(0, 1.0); b.set_ylim(-0.55, len(G) - 0.1); b.set_xticks([0, 0.5, 1.0])
    b.set_xlabel("Density gradient (per km)", labelpad=2)
    rowaxis(b); panel(fig, b, "b", "Gradients flatten", dx=-0.075, dy=0.03)

    c = fig.add_axes([0.84, 0.165, 0.145, 0.69])
    N = sorted(U, key=lambda r: float(r["reversal_rate_v1_least_cloudy"]))
    for i, r in enumerate(N):
        dumbbell(c, i, 100 * float(r["reversal_rate_v1_least_cloudy"]), 100 * float(r["reversal_rate"]))
    leader(c, (100 * float(N[-1]["reversal_rate_v1_least_cloudy"]), len(N) - 1), (15.5, len(N) - 0.35), "least-cloudy", ha="left")
    leader(c, (100 * float(N[-1]["reversal_rate"]), len(N) - 1), (2.5, len(N) - 0.35), "median", ha="left")
    c.set_yticks(range(len(N)), [r["city"].title() for r in N])
    c.set_xlim(0, 26); c.set_ylim(-0.55, len(N) - 0.1); c.set_xticks([0, 10, 20])
    c.set_xlabel("Pixels that flip back (%)", labelpad=2)
    rowaxis(c); panel(fig, c, "c", "Compositing cuts noise", dx=-0.085, dy=0.03)
    fig.savefig(FIG / "urban_area_and_profiles.png", dpi=300)


if __name__ == "__main__":
    style()
    which = sys.argv[2:] or ["atlas", "southwest", "padma", "model", "wealth", "city_maps", "urban_series"]
    M = Maps() if set(which) & {"atlas", "southwest", "padma", "model", "wealth"} else None
    for name in which:
        f = globals()[name]
        f(M) if name in ("atlas", "southwest", "padma", "model", "wealth") else f()
        plt.close("all")
