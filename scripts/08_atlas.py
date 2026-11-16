"""Sub-district (upazila) indicator atlas from committed outputs.

Reads outputs/tables/qse_padma_main.csv (544 upazilas) and GADM level-3
boundaries, and writes docs/atlas/{upazila_indicators.csv, division_summary.csv,
atlas_maps.png, atlas_maps_southwest.png, ATLAS.md}.

Usage: python scripts/08_atlas.py /path/to/gadm41_BGD_3.json
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/atlas"
OUT.mkdir(parents=True, exist_ok=True)
GADM3 = Path(sys.argv[1])

COLS = {
    "GID_3": "gid_3", "NAME_1": "division", "NAME_2": "district", "NAME_3": "upazila",
    "pop": "population_2022", "rwi": "rwi_meta", "built_share": "built_share_wc2021",
    "D_nat": "d_log_tt_reduction",
}


def main():
    m = pd.read_csv(ROOT / "outputs/tables/qse_padma_main.csv").rename(columns=COLS)
    up = gpd.read_file(GADM3)[["GID_3", "geometry"]].rename(columns={"GID_3": "gid_3"})
    g = up.merge(m, on="gid_3", how="inner")
    g["area_km2"] = g.to_crs(epsg=3106).area / 1e6
    g["pop_density_per_km2"] = g.population_2022 / g.area_km2
    g["built_share_wc2021_pct"] = 100 * g.built_share_wc2021
    g["tt_reduction_pct"] = 100 * (1 - np.exp(-g.d_log_tt_reduction))
    g["treated_d_gt_0_1"] = (g.d_log_tt_reduction > 0.1).astype(int)
    g["real_income_sr_pct"] = 100 * (g.V_hat_sr - 1)
    g["wage_sr_pct"] = 100 * (g.w_hat_sr - 1)
    g["population_lr_pct"] = 100 * (g.L_hat_lr - 1)
    g["real_income_lr_pct"] = 100 * (g.V_hat_lr - 1)
    keep = ["gid_3", "division", "district", "upazila", "area_km2", "population_2022", "pop_density_per_km2",
            "rwi_meta", "built_share_wc2021_pct", "d_log_tt_reduction", "tt_reduction_pct", "treated_d_gt_0_1",
            "wage_sr_pct", "real_income_sr_pct", "population_lr_pct", "real_income_lr_pct"]
    t = g[keep].sort_values(["division", "district", "upazila"]).reset_index(drop=True)
    t.round(4).to_csv(OUT / "upazila_indicators.csv", index=False)

    w = lambda d, c: np.average(d[c], weights=d.population_2022)  # noqa: E731
    rows = []
    for div, d in t.groupby("division"):
        rows.append(dict(division=div, upazilas=len(d), population_m=d.population_2022.sum() / 1e6,
                         area_km2=d.area_km2.sum(), rwi_meta=w(d, "rwi_meta"),
                         built_share_wc2021_pct=w(d, "built_share_wc2021_pct"),
                         tt_reduction_pct=w(d, "tt_reduction_pct"),
                         share_pop_treated=d.population_2022[d.treated_d_gt_0_1 == 1].sum() / d.population_2022.sum(),
                         real_income_sr_pct=w(d, "real_income_sr_pct"), population_lr_pct=w(d, "population_lr_pct")))
    ds = pd.DataFrame(rows).sort_values("real_income_sr_pct", ascending=False)
    ds.round(3).to_csv(OUT / "division_summary.csv", index=False)

    # ---- maps
    panels = [
        ("rwi_meta", "Relative Wealth Index (Meta), pop.-weighted mean", "viridis", None),
        ("built_share_wc2021_pct", "Built-up share, WorldCover 2021 (%)", "magma", LogNorm(vmin=0.1, vmax=100)),
        ("pop_density_per_km2", "Population density 2022 (persons/km²)", "cividis", LogNorm(vmin=100, vmax=30000)),
        ("tt_reduction_pct", "Travel-time reduction to Dhaka from the bridge (%)", "YlGnBu", None),
        ("real_income_sr_pct", "Short-run real income change (%)", "RdBu_r", TwoSlopeNorm(vcenter=0)),
        ("population_lr_pct", "Long-run population change (%)", "RdBu_r", TwoSlopeNorm(vcenter=0)),
    ]
    def kw(d, col, norm):
        return dict(norm=norm) if norm is not None else dict(vmin=float(d[col].min()), vmax=float(d[col].max()))

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, (col, title, cmap, norm) in zip(axes.ravel(), panels):
        d = g.copy()
        if isinstance(norm, LogNorm):
            d[col] = d[col].clip(lower=norm.vmin)
        d.plot(col, ax=ax, cmap=cmap, legend=True, legend_kwds=dict(shrink=0.6), edgecolor="none", **kw(d, col, norm))
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Sub-district indicator atlas, Bangladesh (544 upazilas)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "atlas_maps.png", dpi=110)

    sw = g[g.division.isin(["Barisal", "Khulna", "Dhaka"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (col, title, cmap, norm) in zip(axes, [panels[3], panels[4], panels[5]]):
        sw.plot(col, ax=ax, cmap=cmap, legend=True, legend_kwds=dict(shrink=0.7), edgecolor="white", linewidth=0.2,
                **kw(sw, col, norm))
        ax.plot(90.262, 23.446, marker="*", color="red", markersize=12)
        ax.plot(90.40, 23.78, marker="s", color="black", markersize=6)
        ax.set_title(title + " — Dhaka, Khulna, Barisal divisions", fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "atlas_maps_southwest.png", dpi=110)

    # ---- markdown atlas
    def md(df, cols, fmt):
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        body = "".join("| " + " | ".join(fmt(r, c) for c in cols) + " |\n" for _, r in df.iterrows())
        return head + body

    def f(r, c):
        v = r[c]
        if isinstance(v, (float, np.floating)):
            return f"{v:,.0f}" if c in ("population_2022", "area_km2", "pop_density_per_km2") else f"{v:.2f}"
        return str(v)

    top = t.sort_values("real_income_sr_pct", ascending=False).head(20)
    bot = t.sort_values("real_income_sr_pct").head(10)
    big = t.sort_values("population_2022", ascending=False).head(15)
    tcols = ["division", "district", "upazila", "population_2022", "rwi_meta", "built_share_wc2021_pct",
             "tt_reduction_pct", "real_income_sr_pct", "population_lr_pct"]
    text = f"""# Sub-District Indicator Atlas, Bangladesh (544 upazilas)

Companion to *Geospatial Monitoring of Land-Use Change, Sub-District Wealth, and the Padma Bridge in Bangladesh (2018–2026)*. All indicators are produced by the project pipeline from open data; see the technical report for sources and methods. Machine-readable files: `upazila_indicators.csv` (one row per upazila) and `division_summary.csv`.

## Indicator definitions

| Column | Definition | Source / method |
|---|---|---|
| `population_2022` | Resident population | HRSL (Meta/CIESIN) native 30 m pixels summed to 1 km and to upazila; national total scaled to the 2022 census (169.83 million) |
| `area_km2`, `pop_density_per_km2` | Land area and density | GADM 4.1 level-3 polygons, projected area |
| `rwi_meta` | Relative Wealth Index, population-weighted mean of grid points in the upazila | Meta Relative Wealth Index (Chi et al. 2022) |
| `built_share_wc2021_pct` | Share of upazila area classified built-up, % | ESA WorldCover 2021 (10 m), averaged on the 1 km grid |
| `d_log_tt_reduction`, `tt_reduction_pct` | Reduction in modelled road travel time to Dhaka attributable to the Padma Bridge: log difference and % | 500 m least-cost travel-time model (Overture roads, ferries, WorldCover water), with and without the bridge; population-weighted upazila mean |
| `treated_d_gt_0_1` | 1 if the log reduction exceeds 0.1 (about 10%) | Same |
| `wage_sr_pct`, `real_income_sr_pct` | Short-run (no migration) change in nominal wage and real income after the bridge, % | Two-sector quantitative spatial equilibrium model, main parameters (σ = 5, α_N = 0.30, κ = 1) |
| `population_lr_pct`, `real_income_lr_pct` | Long-run (with migration, ε = 3) change in population and real income, % | Same |

Model-based columns are counterfactual simulations under stated parameters, not observed changes; see the sensitivity analysis in the report (national welfare 0.30–0.55% across parameters).

## Division summary (population-weighted)

{md(ds, ["division", "upazilas", "population_m", "rwi_meta", "built_share_wc2021_pct", "tt_reduction_pct", "share_pop_treated", "real_income_sr_pct", "population_lr_pct"], lambda r, c: f"{r[c]:.2f}" if isinstance(r[c], (float, np.floating)) else str(r[c]))}

## Maps

![Atlas maps](atlas_maps.png)

![South-west zoom](atlas_maps_southwest.png)

Red star: Padma Bridge; black square: Dhaka.

## Largest modelled gains (top 20 upazilas by short-run real income change)

{md(top, tcols, f)}

## Largest modelled losses (10 upazilas)

{md(bot, tcols, f)}

## Fifteen most populous upazilas

{md(big, tcols, f)}

## Appendix: all 544 upazilas

Sorted by division, district, upazila. Columns as defined above.

{md(t, tcols, f)}
"""
    (OUT / "ATLAS.md").write_text(text)
    print(ds.round(2).to_string(index=False))
    print(len(t), "upazilas written")


if __name__ == "__main__":
    main()
