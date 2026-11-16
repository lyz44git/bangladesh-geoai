"""Figure 1 of the README: a map of the model-implied income change with three supporting panels.

Drawn from the committed result tables. The map needs the GADM 4.1 level-3 boundaries of Bangladesh
(gadm41_BGD_3.json, from gadm.org), which are not stored in this repository.

Usage: python figures/make_figure1.py [path/to/gadm41_BGD_3.json]   (writes figures/figure1.png)
"""
import csv
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "outputs" / "tables"
GADM3 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/gadm/gadm41_BGD_3.json"
INK, INK2, AXIS, SEA = "#111111", "#5a5a5a", "#333333", "#eef3f6"
BLUE, RED, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#9a9a9a"
# loss (orange) - none (cream) - gain (teal to navy)
INCOME = LinearSegmentedColormap.from_list("income", [(0.0, "#D55E00"), (0.5, "#f6f1e4"), (0.64, "#8fd0c0"),
                                                      (0.82, "#2b83ba"), (1.0, "#0b2a66")])
NORM = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=4)
PLACES = {"Dhaka": (90.41, 23.81), "Khulna": (89.56, 22.82), "Barishal": (90.37, 22.70),
          "Chattogram": (91.83, 22.34), "Rajshahi": (88.60, 24.37), "Sylhet": (91.87, 24.90)}
BRIDGE = (90.262, 23.444)


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white"})


def detach(ax):
    """Axes drawn slightly away from the data, as in journal figures."""
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))


def letter(fig, ax, s, title, x=0.497):
    box = ax.get_position()
    fig.text(x, box.y1 + 0.022, s, fontsize=10, fontweight="bold", va="bottom")
    fig.text(x + 0.022, box.y1 + 0.0235, title, fontsize=7.5, va="bottom")


def leader(ax, xy, xytext, text, **kw):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=6.5, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=AXIS, shrinkA=1.5, shrinkB=2), **kw)


def draw():
    fig = plt.figure(figsize=(7.2, 5.0), dpi=300)

    # ---- a: model-implied short-run real income change by upazila
    U = {r["gid_3"]: r for r in rows(ROOT / "docs" / "atlas" / "upazila_indicators.csv")}
    Q = json.loads((TB / "qse_calibration.json").read_text())["main"]
    g = gpd.read_file(GADM3)[["GID_3", "geometry"]]
    g = g[g.GID_3.isin(U)].copy()
    g["inc"] = [float(U[k]["real_income_sr_pct"]) for k in g.GID_3]
    a = fig.add_axes([0.012, 0.03, 0.465, 0.905])
    a.set_facecolor(SEA)
    g.plot(column="inc", cmap=INCOME, norm=NORM, ax=a, edgecolor="white", linewidth=0.15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = g.geometry.buffer(0.004).union_all().buffer(-0.004)
    gpd.GeoSeries([q.exterior for q in getattr(merged, "geoms", [merged]) if q.area > 0.002]).plot(
        ax=a, color=AXIS, linewidth=0.45)
    for name, (x, y) in PLACES.items():
        a.plot(x, y, marker="o", ms=2.6, mfc="white", mec=INK, mew=0.7)
        left = name in ("Chattogram", "Sylhet")
        a.text(x + (-0.07 if left else 0.07), y + 0.02, name, fontsize=6.5, ha="right" if left else "left",
               va="center", path_effects=[matplotlib.patheffects.withStroke(linewidth=1.6, foreground="white")])
    a.annotate("Padma\nBridge", xy=BRIDGE, xytext=(87.98, 23.3), fontsize=6.5, va="center",
               linespacing=1.15, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, shrinkB=2.5, mutation_scale=6))
    a.plot(*BRIDGE, marker="s", ms=3.4, mfc=INK, mec="white", mew=0.7)
    a.set_xlim(87.9, 92.8); a.set_ylim(20.42, 27.0); a.set_aspect(1 / 0.916)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    cax = a.inset_axes([0.05, 0.075, 0.42, 0.016])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=INCOME), cax=cax, orientation="horizontal",
                      ticks=[-1, 0, 1, 2, 3, 4], extend="max")
    cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.5, labelsize=6.5, pad=1.5)
    cb.ax.set_title("Short-run real income change (%)", fontsize=7, loc="left", pad=3)
    a.text(0.05, 0.03, f'National welfare +{Q["welfare_sr_pct"]:.2f}%', transform=a.transAxes, fontsize=6.5, color=INK2)

    # inset: income change against travel-time reduction, one point per upazila
    ins = a.inset_axes([0.62, 0.835, 0.34, 0.135])
    ins.set_facecolor("none")
    tt = [float(U[k]["tt_reduction_pct"]) for k in g.GID_3]
    ins.scatter(tt, g.inc, c=g.inc, cmap=INCOME, norm=NORM, s=5, lw=0.2, edgecolor="white")
    ins.set_xlabel("Travel time to Dhaka cut (%)", fontsize=6.5, labelpad=1.5)
    ins.set_ylabel("Income (%)", fontsize=6.5, labelpad=1.5)
    ins.set_xticks([0, 35, 70]); ins.set_yticks([0, 4, 8]); ins.tick_params(labelsize=6, length=1.8, pad=1.5)
    detach(ins)
    fig.text(0.012, 0.957, "a", fontsize=10, fontweight="bold", va="bottom")
    fig.text(0.034, 0.9585, f"Who gains from the bridge: spatial equilibrium model, {len(g)} upazilas", fontsize=7.5,
             va="bottom")

    X0, W, H = 0.60, 0.385, 0.20
    b, c, d = (fig.add_axes([X0, y, W, H]) for y in (0.735, 0.41, 0.085))

    # ---- b: event study, naive against controlled, shaded 95% intervals
    E = [r for r in rows(TB / "padma_event_study.csv") if r["sample"].startswith("2020-2026")]
    b.axhline(0, color=AXIS, lw=0.5)
    b.axvline(2022.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
    b.text(2022.56, -1.85, "bridge opens", fontsize=6.5, color=INK2, va="center")
    series = {}
    for spec, col in (("(1)", RED), ("(2)", BLUE)):
        pts = sorted([(int(r["year"]), float(r["coef"]), float(r["ci_lo"]), float(r["ci_hi"]))
                      for r in E if r["spec"].startswith(spec)] + [(2022, 0.0, 0.0, 0.0)])
        x = [p[0] for p in pts]
        b.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.16, lw=0)
        b.plot(x, [p[1] for p in pts], color=col, lw=1.2, marker="o", ms=3, mfc="white", mew=1.0)
        series[spec] = pts
    leader(b, (2021, series["(1)"][1][1]), (2020.05, -1.45), "Naive: cell and\nyear fixed effects", linespacing=1.1)
    leader(b, (2024, series["(2)"][-3][1]), (2022.65, 0.56), "Baseline-density trends controlled", ha="left")
    b.set_xlim(2019.85, 2026.15); b.set_ylim(-2.0, 0.7); b.set_xticks(range(2020, 2027)); b.set_yticks([-2, -1, 0])
    b.set_ylabel("Built-up share\n(pp per log point)")
    detach(b); letter(fig, b, "b", "No detectable effect on built-up land, 34,626 cells")

    # ---- c: wealth prediction by feature set, spatial-block against random cross-validation
    Wr = [r for r in rows(TB / "wealth_prediction_results.csv") if r["target"] == "rwi"]
    names = {"context only: pop+travel time (gbm)": "Pop. + travel time", "landcover (gbm)": "Land cover",
             "s2_stats+landcover (gbm)": "S2 + land cover", "imagery + context (gbm)": "Imagery + context"}
    sp = {r["features"]: float(r["r2_upazila"]) for r in Wr if r["cv"].startswith("spatial")}
    rd = {r["features"]: float(r["r2_upazila"]) for r in Wr if r["cv"] == "random"}
    order = sorted(names, key=lambda k: sp[k])
    for i, k in enumerate(order):
        c.plot([sp[k], rd[k]], [i, i], color="#d9dde1", lw=2.6, solid_capstyle="round", zorder=1)
        c.plot(rd[k], i, marker="o", ms=4.4, mfc="white", mec=GREY, mew=1.0, zorder=2)
        c.plot(sp[k], i, marker="o", ms=5, color=BLUE, zorder=3)
        c.text(sp[k] - 0.016, i, f"{sp[k]:.2f}", ha="right", va="center", fontsize=6.5,
               fontweight="bold" if i == len(order) - 1 else "normal")
    top = len(order) - 1
    leader(c, (sp[order[0]], 0), (0.52, 0.95), "50 km spatial-\nblock CV", linespacing=1.1, ha="left")
    leader(c, (rd[order[0]], 0), (0.52, 0.0), "random CV", ha="left")
    c.set_yticks(range(len(order)), [names[k] for k in order]); c.tick_params(axis="y", length=0)
    c.set_xlim(0.38, 0.95); c.set_ylim(-0.55, top + 0.55); c.set_xticks([0.4, 0.6, 0.8])
    c.set_xlabel("Out-of-sample R², upazila wealth", labelpad=2)
    c.spines["left"].set_visible(False); c.spines["bottom"].set_position(("outward", 3))
    letter(fig, c, "c", "Imagery predicts wealth out of area; random CV overstates it")

    # ---- d: built-up growth by city and where the new building is
    U0 = sorted(rows(TB / "urban_expansion_summary.csv"), key=lambda r: float(r["growth_pct"]))
    parts = (("new_built_edge", "urban edge", BLUE), ("new_built_infill", "infill", "#56B4E9"),
             ("new_built_outlying", "outlying", "#c9ced3"))
    for i, r in enumerate(U0):
        gr, left = float(r["growth_pct"]), 0.0
        for key, name, col in parts:
            w = gr * float(r[key])
            d.barh(i, w, left=left, height=0.66, color=col, edgecolor="white", lw=0.7)
            if i == len(U0) - 1:
                d.text(left + w / 2, i + 0.62, name, ha="center", va="bottom", fontsize=6.3, color=INK2)
            left += w
        d.text(gr + 1.2, i, f"+{gr:.0f}%", va="center", fontsize=6.5)
    d.set_yticks(range(len(U0)), [r["city"].title() for r in U0]); d.tick_params(axis="y", length=0)
    d.set_xlim(0, 82); d.set_ylim(-0.6, len(U0) + 0.25); d.set_xticks([0, 20, 40, 60, 80])
    d.set_xlabel("Growth in built-up area, 2020–2026 (%)", labelpad=2)
    d.spines["left"].set_visible(False); d.spines["bottom"].set_position(("outward", 3))
    letter(fig, d, "d", "Cities grew 27–72%, mostly at the urban edge")
    return fig


if __name__ == "__main__":
    import matplotlib.patheffects
    style()
    draw().savefig(Path(__file__).parent / "figure1.png", dpi=300)
