"""Figure 1 of the README: the four steps of the study, measurement (a), description (b), evaluation (c) and
interpretation (d), with the model map as the large panel.

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


def letter(fig, ax, s, title, x=0.012):
    box = ax.get_position()
    fig.text(x, box.y1 + 0.022, s, fontsize=8, fontweight="bold", va="baseline")
    fig.text(x + 0.016, box.y1 + 0.022, title, fontsize=7.5, va="baseline")


def key(ax, entries, loc="upper right", anchor=(1.0, 1.0), vertical=False, face="#f1f3f5", fontsize=6.3):
    """Legend chip: a rounded, borderless box holding (handle, label) pairs.

    A handle is a dict: {"color": c} draws a filled marker; "mfc"/"mec" style an open one;
    {"kind": "line"} draws a line sample and {"kind": "patch"} a square. Anchored in axes coordinates.
    """
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
    from matplotlib.patches import Rectangle

    items = []
    for h, label in entries:
        kind, col = h.get("kind", "marker"), h["color"]
        da = DrawingArea(11 if kind == "line" else 8, 7, 0, 0)
        if kind == "patch":
            da.add_artist(Rectangle((1.2, 0.7), 5.6, 5.6, fc=col, ec="none"))
        elif kind == "line":
            da.add_artist(Line2D([0.5, 10.5], [3.5, 3.5], color=col, lw=h.get("lw", 1.3), ls=h.get("ls", "-"),
                                 solid_capstyle="round"))
        elif kind == "bar":
            da.add_artist(Line2D([4, 4], [0.3, 6.7], color=col, lw=1.3, solid_capstyle="butt"))
        else:
            da.add_artist(Line2D([4], [3.5], marker=h.get("marker", "o"), ms=h.get("ms", 4.2), mfc=h.get("mfc", col),
                                 mec=h.get("mec", col), mew=h.get("mew", 1.0), ls="none"))
        items.append(HPacker(children=[da, TextArea(label, textprops=dict(fontsize=fontsize, color=INK))],
                             align="center", pad=0, sep=1.5))
    pack = (VPacker(children=items, align="left", pad=0, sep=2.5) if vertical
            else HPacker(children=items, align="center", pad=0, sep=7))
    box = AnchoredOffsetbox(loc=loc, child=pack, pad=0.45, borderpad=0, frameon=True, bbox_to_anchor=anchor,
                            bbox_transform=ax.transAxes, prop=dict(size=fontsize))
    box.patch.set(boxstyle="round,pad=0,rounding_size=0.55", facecolor=face, edgecolor="none")
    box.set_zorder(6)
    ax.add_artist(box)
    return box


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
    a = fig.add_axes([0.515, 0.03, 0.475, 0.905])
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
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel(""); a.set_ylabel("")
    for s in a.spines.values():
        s.set_visible(False)
    cax = a.inset_axes([0.05, 0.10, 0.42, 0.016])
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
    fig.text(0.515, 0.957, "d", fontsize=8, fontweight="bold", va="baseline")
    fig.text(0.531, 0.957, f"Model-implied welfare gain of {Q['welfare_sr_pct']:.2f}%, concentrated in the southwest", fontsize=7.5,
             va="baseline")

    X0, W, H = 0.135, 0.33, 0.20
    c, d, b = (fig.add_axes([X0, y, W, H]) for y in (0.735, 0.41, 0.085))

    # ---- b: event study, naive against controlled, shaded 95% intervals
    E = [r for r in rows(TB / "padma_event_study.csv") if r["sample"].startswith("2020-2026")]
    b.axhline(0, color=AXIS, lw=0.5)
    b.axvline(2022.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
    b.text(2022.56, 0.56, "bridge opens", fontsize=6.5, color=INK2, va="center")
    series = {}
    for spec, col in (("(1)", RED), ("(2)", BLUE)):
        pts = sorted([(int(r["year"]), float(r["coef"]), float(r["ci_lo"]), float(r["ci_hi"]))
                      for r in E if r["spec"].startswith(spec)] + [(2022, 0.0, 0.0, 0.0)])
        x = [p[0] for p in pts]
        b.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.16, lw=0)
        b.plot(x, [p[1] for p in pts], color=col, lw=1.2, marker="o", ms=3, mfc="white", mew=1.0)
        series[spec] = pts
    key(b, [({"kind": "line", "color": RED}, "cell and year fixed effects only"),
            ({"kind": "line", "color": BLUE}, "trends by baseline density controlled")], loc="lower left",
        anchor=(0.03, 0.03), vertical=True)
    b.set_xlim(2019.85, 2026.15); b.set_ylim(-2.0, 0.7); b.set_xticks(range(2020, 2027)); b.set_yticks([-2, -1, 0])
    b.set_ylabel("Built-up share\n(pp per log point)")
    detach(b); letter(fig, b, "c", "No detectable effect of the bridge (power 8–23%)")

    # ---- a: accuracy of the two measures on held-out areas: built-up land (F1) and upazila wealth (R2)
    Bm = {r["model"]: float(r["f1"]) for r in rows(TB / "builtup_model_comparison_cities.csv") if r["aoi"] == "pooled"}
    Wr = [r for r in rows(TB / "wealth_prediction_results.csv") if r["target"] == "rwi"]
    sp = {r["features"]: float(r["r2_upazila"]) for r in Wr if r["cv"].startswith("spatial")}
    rd = {r["features"]: float(r["r2_upazila"]) for r in Wr if r["cv"] == "random"}
    best, ctx = "imagery + context (gbm)", "context only: pop+travel time (gbm)"
    #        y, label, held-out value, random-fold value, emphasised
    spec = [(4.0, "U-Net", Bm["unet"], None, True), (3.0, "Gradient boosting", Bm["gbm"], None, False),
            (1.0, "Imagery + context", sp[best], rd[best], True), (0.0, "Pop. + travel time", sp[ctx], rd[ctx], False)]
    for y, name, v, r, strong in spec:
        if r is not None:
            c.plot([v, r], [y, y], color="#d9dde1", lw=2.6, solid_capstyle="round", zorder=1)
            c.plot(r, y, marker="o", ms=4.4, mfc="white", mec=GREY, mew=1.0, zorder=2)
        c.plot(v, y, marker="o", ms=5, color=BLUE if strong else "#8fb8d6", zorder=3)
        c.text(v - 0.016, y, f"{v:.2f}", ha="right", va="center", fontsize=6.5, fontweight="bold" if strong else "normal")
    for y, head in ((4.72, "Built-up land, F1 against WorldCover"), (1.72, "Upazila wealth, R² against the wealth index")):
        c.text(0.385, y, head, fontsize=6.3, color=INK2, va="center")
    key(c, [({"color": BLUE}, "held-out areas"), ({"color": GREY, "mfc": "white"}, "random folds")],
        loc="lower right", anchor=(1.0, 0.0))
    c.set_yticks([y for y, *_ in spec], [n for _, n, *_ in spec]); c.tick_params(axis="y", length=0)
    c.set_xlim(0.38, 0.95); c.set_ylim(-0.6, 5.15); c.set_xticks([0.4, 0.6, 0.8])
    c.set_xlabel("Accuracy on held-out areas", labelpad=2)
    c.spines["left"].set_visible(False); c.spines["bottom"].set_position(("outward", 3))
    letter(fig, c, "a", "Out-of-area accuracy: built-up F1 = 0.80, wealth R² = 0.83")

    # ---- d: built-up growth by city and where the new building is
    U0 = sorted(rows(TB / "urban_expansion_summary.csv"), key=lambda r: float(r["growth_pct"]))
    parts = (("new_built_edge", "urban edge", BLUE), ("new_built_infill", "infill", "#56B4E9"),
             ("new_built_outlying", "outlying", "#c9ced3"))
    for i, r in enumerate(U0):
        gr, left = float(r["growth_pct"]), 0.0
        for field, name, col in parts:
            w = gr * float(r[field])
            d.barh(i, w, left=left, height=0.66, color=col, edgecolor="white", lw=0.7)
            left += w
        d.text(gr + 1.2, i, f"+{gr:.0f}%", va="center", fontsize=6.5)
    d.set_yticks(range(len(U0)), [r["city"].title() for r in U0]); d.tick_params(axis="y", length=0)
    key(d, [({"kind": "patch", "color": col}, name) for _, name, col in parts], loc="center right", anchor=(1.0, 0.42),
        vertical=True)
    d.set_xlim(0, 82); d.set_ylim(-0.6, len(U0) - 0.4); d.set_xticks([0, 20, 40, 60, 80])
    d.set_xlabel("Growth in built-up area, 2020–2026 (%)", labelpad=2)
    d.spines["left"].set_visible(False); d.spines["bottom"].set_position(("outward", 3))
    letter(fig, d, "b", "Built-up area grew 27–72% in six cities, mostly at the edge")
    return fig


if __name__ == "__main__":
    import matplotlib.patheffects
    style()
    draw().savefig(Path(__file__).parent / "figure1.png", dpi=300)
