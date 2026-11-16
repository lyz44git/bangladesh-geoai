"""Urban expansion of six Bangladeshi cities from annual built-up maps.

Main period 2020-2026 uses Sentinel-2 Collection 1 only (one processing
baseline). 2018-2019 come from the older L2A archive and are shown separately.

Outputs
  outputs/tables/urban_expansion_summary.csv   area, growth, expansion types, density gradients
  outputs/tables/urban_built_area_by_year.csv
  outputs/figures/urban_*.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage import measure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoai.builtup import read_prob  # noqa: E402
from geoai.grid import lonlat_to_tm  # noqa: E402

CFG = json.loads((ROOT / "configs/aois.json").read_text())
PRED, COMP = ROOT / "data/builtup", ROOT / "data/composites"
TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"
YEARS = CFG["years"]
MAIN = [y for y in YEARS if y >= 2020]
T0, T1 = MAIN[0], MAIN[-1]


def load_city(name, pred_dir=None):
    pred_dir = pred_dir or PRED
    probs, clear = [], []
    for y in YEARS:
        probs.append(read_prob(pred_dir / f"city_{name}_{y}.tif"))
        with rasterio.open(COMP / f"city_{name}_{y}.tif") as src:
            clear.append(src.read(src.count))
            tr, res = src.transform, src.res[0]
    return np.stack(probs), np.stack(clear), tr, res


def temporal_smooth(P, years):
    """Centred 3-year median of probabilities within the main period (edges: 2-year median)."""
    S = P.copy()
    idx = [years.index(y) for y in MAIN]
    for k, i in enumerate(idx):
        lo, hi = idx[max(0, k - 1)], idx[min(len(idx) - 1, k + 1)]
        S[i] = np.nanmedian(P[[lo, i, hi]], axis=0)
    return S


def consistency(P, valid, years):
    """Temporal-consistency diagnostics of raw annual maps over the main period.

    reversal_rate: share of pixels built in year t that are classified non-built
    in t+1 (true reversals are rare, so this is mostly classification noise).
    trend_resid_sd: SD of residuals of log built area around a linear trend.
    """
    idx = [years.index(y) for y in MAIN]
    B = [(P[i] >= 0.5) & valid for i in idx]
    rev = [float((B[k] & ~B[k + 1]).sum() / max(B[k].sum(), 1)) for k in range(len(B) - 1)]
    la = np.log([b.sum() for b in B])
    t = np.array(MAIN, float)
    resid = la - np.polyval(np.polyfit(t, la, 1), t)
    return dict(reversal_rate=float(np.mean(rev)), trend_resid_sd=float(resid.std(ddof=2)))


def lei_shares(b0, b1, ring_px=2):
    """Landscape Expansion Index (Liu et al. 2010) of new built patches between two dates.

    LEI = 100 * (built-at-t0 share of a ring around the new patch):
    infill > 50, edge-expansion 0-50, outlying = 0. Shares are by area.
    """
    new = b1 & ~b0
    lab = measure.label(new, connectivity=2)
    out = {"infill": 0, "edge": 0, "outlying": 0}
    if lab.max() == 0:
        return {k: np.nan for k in out}
    objs = ndimage.find_objects(lab)
    for i, sl in enumerate(objs, 1):
        r0, r1 = max(sl[0].start - ring_px, 0), sl[0].stop + ring_px
        c0, c1 = max(sl[1].start - ring_px, 0), sl[1].stop + ring_px
        patch = lab[r0:r1, c0:c1] == i
        ring = ndimage.binary_dilation(patch, iterations=ring_px) & ~patch
        lei = 100 * b0[r0:r1, c0:c1][ring].mean() if ring.any() else 0
        a = patch.sum()
        out["infill" if lei > 50 else "edge" if lei > 0 else "outlying"] += a
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def radial_profile(b, tr, res, lon, lat, valid, max_km=None, step_km=1.0):
    x0, y0 = lonlat_to_tm(lon, lat)
    h, w = b.shape
    xs = tr.c + (np.arange(w) + 0.5) * res
    ys = tr.f - (np.arange(h) + 0.5) * res
    d = np.hypot(xs[None, :] - x0, ys[:, None] - y0) / 1000
    max_km = max_km or d.max()
    bins = np.arange(0, max_km + step_km, step_km)
    k = np.digitize(d, bins) - 1
    share = np.array([b[(k == i) & valid].mean() if ((k == i) & valid).any() else np.nan for i in range(len(bins) - 1)])
    return bins[:-1] + step_km / 2, share


def gradient(r, s, r_min=1.0):
    """Negative-exponential density gradient: log(share) = a - b r (fit where share > 1%)."""
    ok = np.isfinite(s) & (s > 0.01) & (r >= r_min)
    if ok.sum() < 4:
        return np.nan
    return -np.polyfit(r[ok], np.log(s[ok]), 1)[0]


def main():
    rows, area_rows, profiles = [], [], {}
    fig_map, axes_map = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, c) in zip(axes_map.ravel(), CFG["cities"].items()):
        P, CL, tr, res = load_city(name)
        main_idx = [YEARS.index(y) for y in MAIN]
        # Common footprint: observed in every main-period year. (The older 2018-2019
        # composites have gaps, e.g. 18% no data over Khulna in 2018, and must not
        # shrink the main-period footprint.)
        valid = np.isfinite(P[main_idx]).all(0) & (CL[main_idx] > 0).all(0)
        S = temporal_smooth(P, YEARS)
        px_km2 = res * res / 1e6
        for i, y in enumerate(YEARS):
            v = valid if y in MAIN else valid & np.isfinite(P[i]) & (CL[i] > 0)
            area_rows.append(dict(city=name, year=y, built_km2_raw=float(((P[i] >= 0.5) & v).sum() * px_km2),
                                  built_km2_smoothed=float(((S[i] >= 0.5) & valid).sum() * px_km2) if y in MAIN else np.nan,
                                  footprint_km2=float(v.sum() * px_km2),
                                  median_clear_obs=float(np.median(CL[i][v]))))
        b0 = (S[YEARS.index(T0)] >= 0.5) & valid
        b1 = (S[YEARS.index(T1)] >= 0.5) & valid
        # Built-up is treated as persistent: a pixel built at T0 stays built at T1.
        b1 = b1 | b0
        a0, a1 = b0.sum() * px_km2, b1.sum() * px_km2
        lei = lei_shares(b0, b1)
        r, s0 = radial_profile(b0, tr, res, c["lon"], c["lat"], valid, max_km=c["half_km"])
        _, s1 = radial_profile(b1, tr, res, c["lon"], c["lat"], valid, max_km=c["half_km"])
        profiles[name] = (r, s0, s1)
        rows.append(dict(city=name, footprint_km2=valid.sum() * px_km2, built_km2_t0=a0, built_km2_t1=a1,
                         t0=T0, t1=T1, growth_pct=100 * (a1 / a0 - 1),
                         cagr_pct=100 * ((a1 / a0) ** (1 / (T1 - T0)) - 1),
                         new_built_infill=lei["infill"], new_built_edge=lei["edge"], new_built_outlying=lei["outlying"],
                         gradient_b_t0=gradient(r, s0), gradient_b_t1=gradient(r, s1)))
        rows[-1].update(consistency(P, valid, YEARS))
        v1 = ROOT / "data/v1_least_cloudy/builtup"
        if (v1 / f"city_{name}_{T1}.tif").exists():
            P1, _, _, _ = load_city(name, v1)
            valid1 = np.isfinite(P1[main_idx]).all(0) & valid
            for k, v in consistency(P1, valid1, YEARS).items():
                rows[-1][f"{k}_v1_least_cloudy"] = v
        # change map
        rgb = np.ones(b0.shape + (3,))
        rgb[b0] = (0.35, 0.35, 0.35)
        rgb[b1 & ~b0] = (0.85, 0.15, 0.1)
        rgb[~valid] = (0.8, 0.85, 0.95)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{name.title()}: built {T0} (grey), new {T0}-{T1} (red)", fontsize=10)
        ax.set_axis_off()
    fig_map.tight_layout()
    fig_map.savefig(FIG / "urban_change_maps.png", dpi=110)
    summ = pd.DataFrame(rows)
    summ.to_csv(TAB / "urban_expansion_summary.csv", index=False)
    area = pd.DataFrame(area_rows)
    area.to_csv(TAB / "urban_built_area_by_year.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for name in CFG["cities"]:
        a = area[area.city == name].set_index("year")
        base = a.loc[T0, "built_km2_smoothed"]
        ax[0].plot(MAIN, 100 * a.loc[MAIN, "built_km2_smoothed"] / base, marker="o", label=name.title())
        ax[0].plot([2018, 2019, 2020], 100 * a.loc[[2018, 2019, 2020], "built_km2_raw"] / a.loc[2020, "built_km2_raw"],
                   ls=":", color=ax[0].lines[-1].get_color())
    ax[0].set_title(f"Built-up area index ({T0}=100); dotted = older L2A archive, raw")
    ax[0].set_xlabel("dry season ending in year")
    ax[0].legend(fontsize=8)
    for name, (r, s0, s1) in profiles.items():
        l, = ax[1].plot(r, s1, label=f"{name.title()} {T1}")
        ax[1].plot(r, s0, ls="--", color=l.get_color())
    ax[1].set_yscale("log")
    ax[1].set_xlabel("distance from centre (km)")
    ax[1].set_ylabel("built-up share")
    ax[1].set_title(f"Radial built-up profiles: {T0} dashed, {T1} solid")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "urban_area_and_profiles.png", dpi=110)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
