"""Figure 1 of the README: one panel per part, drawn from the committed result tables.

Usage: python figures/make_figure1.py   (writes figures/figure1.png)
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
T = dict(bg="#ffffff", ink="#111111", ink2="#555555", axis="#333333", grid="#e6e6e6", band="#e9eff5",
         blue="#0072B2", red="#D55E00", green="#009E73", grey="#8c8c8c")
FIGSIZE = (7.2, 5.6)
X0, W, H = (0.115, 0.625), 0.355, 0.30          # two equal columns
Y0 = (0.60, 0.085)                              # two equal rows


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def style(t):
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["ink"], "axes.labelcolor": t["ink"], "axes.edgecolor": t["axis"],
        "xtick.color": t["axis"], "ytick.color": t["axis"], "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2, "legend.frameon": False})


def grid(fig):
    """Four axes of equal size on a 2 x 2 grid."""
    return [fig.add_axes([x, y, W, H]) for y in Y0 for x in X0]


def panel(fig, ax, letter, title, legend_rows=1):
    """Panel letter and bold title above the legend strip; the strip holds `legend_rows` rows in this figure row."""
    box = ax.get_position()
    y = box.y1 + (7 + 10.5 * legend_rows) / 72 / fig.get_figheight()
    fig.text(box.x0 - 0.105, y, letter, fontsize=11.5, fontweight="bold", va="bottom")
    fig.text(box.x0, y, title, fontsize=8.5, fontweight="bold", va="bottom")


def legend(ax, ncol, handles=None, labels=None):
    """Legend in a strip between the title and the axes, clear of the data."""
    args = (handles, labels) if handles is not None else ()
    return ax.legend(*args, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.35, borderpad=0,
                     handlelength=1.3, handletextpad=0.5, columnspacing=1.4, labelspacing=0.35)


def draw(t):
    TB = ROOT / "outputs" / "tables"
    fig = plt.figure(figsize=FIGSIZE, dpi=300)
    a, b, c, d = grid(fig)

    # a: built-up growth 2020-2026 by city, split by where the new building is
    U0 = sorted(rows(TB / "urban_expansion_summary.csv"), key=lambda r: float(r["growth_pct"]))
    parts = (("new_built_edge", "Urban edge", t["blue"]), ("new_built_infill", "Infill", t["green"]),
             ("new_built_outlying", "Outlying", t["grey"]))
    for i, r in enumerate(U0):
        g, left = float(r["growth_pct"]), 0.0
        for key, name, colr in parts:
            w = g * float(r[key])
            a.barh(i, w, left=left, height=0.6, color=colr, edgecolor=t["bg"], lw=0.8, label=name if i == 0 else None)
            left += w
        a.text(g + 1.2, i, f"+{g:.0f}%", va="center")
    a.set_yticks(range(len(U0)), [r["city"].title() for r in U0]); a.tick_params(axis="y", length=0)
    a.set_xlim(0, 84); a.set_ylim(-0.6, len(U0) - 0.4)
    a.set_xlabel("Growth in built-up area, 2020–2026 (%)")
    legend(a, 3)
    panel(fig, a, "a", "Urban expansion by location of new building")

    # b: accuracy of wealth prediction by feature set, spatial-block against random cross-validation
    Wr = [r for r in rows(TB / "wealth_prediction_results.csv") if r["target"] == "rwi"]
    names = {"context only: pop+travel time (gbm)": "Pop. + travel time", "s2_stats (gbm)": "S2 statistics",
             "mosaiks (ridge)": "Random conv. feat.", "landcover (gbm)": "Land cover",
             "s2_stats+landcover (gbm)": "S2 + land cover", "imagery all: s2+lc+mosaiks50 (gbm)": "All imagery",
             "imagery + context (gbm)": "Imagery + context"}
    sp = {r["features"]: float(r["r2_upazila"]) for r in Wr if r["cv"].startswith("spatial")}
    rd = {r["features"]: float(r["r2_upazila"]) for r in Wr if r["cv"] == "random"}
    order = sorted(names, key=lambda k: sp[k])
    for i, k in enumerate(order):
        b.plot([sp[k], rd[k]], [i, i], color=t["grid"], lw=2.2, solid_capstyle="round", zorder=1)
    b.plot([sp[k] for k in order], range(len(order)), ls="none", marker="o", ms=4.6, color=t["blue"],
           label="Spatial-block CV (50 km)")
    b.plot([rd[k] for k in order], range(len(order)), ls="none", marker="o", ms=4.2, mfc=t["bg"], mec=t["grey"],
           mew=1.0, label="Random CV")
    b.text(sp[order[-1]] - 0.02, len(order) - 1, f"{sp[order[-1]]:.2f}", ha="right", va="center", fontweight="bold")
    b.set_yticks(range(len(order)), [names[k] for k in order]); b.tick_params(axis="y", length=0)
    b.set_xlim(0.4, 0.96); b.set_ylim(-0.6, len(order) - 0.4)
    b.set_xlabel("Out-of-sample R², upazila means")
    legend(b, 2)
    panel(fig, b, "b", "Wealth prediction by feature set")

    # c: event study of the Padma Bridge, naive against controlled
    E = [r for r in rows(TB / "padma_event_study.csv") if r["sample"].startswith("2020-2026")]
    c.axhline(0, color=t["axis"], lw=0.6)
    c.axvline(2022.5, color=t["grey"], lw=0.8, ls=(0, (4, 3)))
    c.text(2022.58, -1.9, "bridge opens", color=t["ink2"], fontsize=7, va="center")
    for spec, col, name, dx in (("(1)", t["red"], "Cell + year FE", -0.07),
                                 ("(2)", t["blue"], "+ baseline-density trends", 0.07)):
        pts = sorted([(int(r["year"]), float(r["coef"]), float(r["ci_lo"]), float(r["ci_hi"]))
                      for r in E if r["spec"].startswith(spec)] + [(2022, 0.0, 0.0, 0.0)])
        c.errorbar([p[0] + dx for p in pts], [p[1] for p in pts],
                   yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], color=col, marker="o", ms=3.6,
                   lw=1.1, elinewidth=0.9, capsize=0, label=name)
    c.set_xlim(2019.6, 2026.4); c.set_ylim(-2.05, 0.55); c.set_xticks(range(2020, 2027))
    c.set_ylabel("Built-up share (pp per log point)")
    legend(c, 2)
    panel(fig, c, "c", "Padma Bridge and built-up land, 34,626 cells")

    # d: model-implied real income change against travel-time reduction, one point per upazila
    U = rows(ROOT / "docs" / "atlas" / "upazila_indicators.csv")
    Q = json.loads((TB / "qse_calibration.json").read_text())["main"]
    for treated, col, name in ((1, t["blue"], "Travel time to Dhaka cut > 10%"), (0, t["grey"], "Other upazilas")):
        g = [r for r in U if int(r["treated_d_gt_0_1"]) == treated]
        d.scatter([float(r["tt_reduction_pct"]) for r in g], [float(r["real_income_sr_pct"]) for r in g],
                  s=[max(3.0, float(r["population_2022"]) / 22000) for r in g], color=col, alpha=0.55, lw=0, label=name)
    d.axhline(0, color=t["axis"], lw=0.6)
    d.axhline(Q["welfare_sr_pct"], color=t["ink"], lw=0.8, ls=(0, (4, 3)))
    d.text(73.5, Q["welfare_sr_pct"] + 0.15, f'national welfare, +{Q["welfare_sr_pct"]:.2f}%', fontsize=7, ha="right")
    d.set_xlim(-3, 74); d.set_ylim(-0.8, 9.2)
    d.set_xlabel("Travel-time reduction to Dhaka (%)"); d.set_ylabel("Short-run real income change (%)")
    lg = legend(d, 2)
    for h in lg.legend_handles:
        h.set_sizes([20]); h.set_alpha(0.9)
    panel(fig, d, "d", f"Incidence across {len(U)} upazilas (spatial model)")
    return fig


if __name__ == "__main__":
    style(T)
    fig = draw(T)
    fig.savefig(Path(__file__).parent / "figure1.png", dpi=300)
