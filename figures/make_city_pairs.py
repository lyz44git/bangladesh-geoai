"""Figure 3 of the README: each of the six cities as Sentinel-2 sees it, next to the built-up land mapped from it.

Left of each pair, the natural-color dry-season composite for the end year of the urban-expansion analysis, with one
reflectance stretch shared by all six cities. Right, built-up land in the start year and land added since, from the
U-Net maps, drawn as in outputs/figures/urban_change_maps.png (the same maps at full size).

Needs the pipeline rasters in data/composites and data/builtup, or in the folder named by GEOAI_DATA.

Usage: python figures/make_city_pairs.py     (writes outputs/figures/urban_change_pairs.jpg)
"""
import json
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from make_report_figures import (C_BG, C_NEW, C_OLD, DATA, FIG, INK, INK2, ROOT, SEA, TB, city_rasters, key, rows,  # noqa: E402
                                 style)

REFL_LO, REFL_RANGE, GAMMA, BRIGHT = 0.02, 0.16, 0.7, 0.92   # reflectance stretch shared by all six cities
# layout in inches: page width, outer margin, gap inside a pair, gap between pairs, header, top band, row gap, bottom
WIDTH, MARGIN, GAP_IN, GAP_PAIR, HEAD, TOP, GAP_ROW, BOTTOM = 7.2, 0.06, 0.05, 0.22, 0.24, 0.20, 0.07, 0.05


def natural_color(path):
    """Red, green and blue reflectance as a display image; pixels without a clear observation in mid-grey."""
    with rasterio.open(path) as src:
        rgb = src.read([3, 2, 1]).astype("float32") / 1e4
    rgb[rgb == 0] = np.nan
    img = np.clip((np.moveaxis(rgb, 0, -1) - REFL_LO) / REFL_RANGE, 0, 1) ** GAMMA
    return np.nan_to_num(img, nan=0.3) * BRIGHT


def built_up_map(d):
    """Built-up land at the start and land added since, in the colors of urban_change_maps.png."""
    col = {k: np.array(matplotlib.colors.to_rgb(v)) for k, v in (("old", C_OLD), ("new", C_NEW), ("bg", C_BG), ("sea", SEA))}
    img = np.ones(d["b0"].shape + (3,)) * col["bg"]
    img[~d["valid"]] = col["sea"]
    img[d["b0"]] = col["old"]
    img[d["b1"] & ~d["b0"]] = col["new"]
    return img


def draw():
    R = city_rasters()
    if R is None:
        sys.exit("The pipeline rasters are needed: data/composites and data/builtup, or GEOAI_DATA")
    m = runpy.run_path(str(ROOT / "scripts" / "03_urban_expansion.py"), run_name="figures")
    t0, t1 = m["T0"], m["T1"]
    cfg = json.loads((ROOT / "configs" / "aois.json").read_text())["cities"]
    U = {r["city"]: r for r in rows(TB / "urban_expansion_summary.csv")}
    w = (WIDTH - 2 * MARGIN - 2 * GAP_IN - GAP_PAIR) / 4
    n_rows = (len(cfg) + 1) // 2
    height = TOP + n_rows * (HEAD + w) + (n_rows - 1) * GAP_ROW + BOTTOM
    fig = plt.figure(figsize=(WIDTH, height), dpi=300)
    halo = [pe.withStroke(linewidth=1.6, foreground=C_BG)]
    for i, name in enumerate(cfg):
        d, row, col = R[name], i // 2, i % 2
        n = d["b0"].shape[0]
        x0 = MARGIN + col * (2 * w + GAP_IN + GAP_PAIR)
        y0 = height - TOP - (row + 1) * (HEAD + w) - row * GAP_ROW
        images = (natural_color(DATA / "composites" / f"city_{name}_{t1}.tif"), built_up_map(d))
        for k, img in enumerate(images):
            xk = x0 + k * (w + GAP_IN)
            ax = fig.add_axes([xk / WIDTH, y0 / height, w / WIDTH, w / height])
            ax.imshow(img, interpolation="antialiased", extent=(0, n, n, 0))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(k == 1); sp.set_color("#c9ced3"); sp.set_linewidth(0.5)
            ax.set_xlim(0, n); ax.set_ylim(n, 0)
            if row == 0:  # column headings, once at the top
                fig.text((xk + w / 2) / WIDTH, (height - TOP + 0.075) / height,
                         f"Sentinel-2 composite, {t1}" if k == 0 else "Built-up land mapped from it", fontsize=6.5,
                         color=INK2, ha="center", va="baseline")
        px = n * 10 / (2 * cfg[name]["half_km"])
        ax.plot([0.953 * n - px, 0.953 * n], [0.965 * n] * 2, color=INK, lw=1.0, solid_capstyle="butt")
        ax.text(0.953 * n - px / 2, 0.95 * n, "10 km", ha="center", va="bottom", fontsize=6, path_effects=halo)
        if i == 1:
            key(ax, [({"kind": "patch", "color": C_OLD}, f"built-up in {t0}"),
                     ({"kind": "patch", "color": C_NEW}, f"added {t0}–{t1}")],
                loc="upper right", anchor=(0.97, 0.97), vertical=True, face="white")
        r, yt = U[name], (y0 + w + 0.055) / height
        fig.text(x0 / WIDTH, yt, "abcdef"[i], fontsize=8, fontweight="bold", va="baseline")
        fig.text((x0 + 0.115) / WIDTH, yt, f'{name.title()}  +{float(r["growth_pct"]):.0f}%', fontsize=7.5, va="baseline")
        fig.text((x0 + 2 * w + GAP_IN) / WIDTH, yt, f'{float(r["built_km2_t0"]):.0f} → {float(r["built_km2_t1"]):.0f} km²',
                 fontsize=6.5, va="baseline", ha="right", color=INK2)
    return fig


if __name__ == "__main__":
    style()
    draw().savefig(FIG / "urban_change_pairs.jpg", dpi=300, pil_kwargs={"quality": 92})
