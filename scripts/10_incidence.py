"""Who gains in the spatial model: incidence of the bridge by wealth, by treatment status and by sign.

Reads the committed model table only (outputs/tables/qse_padma_main.csv) and writes
outputs/tables/qse_incidence.json. Wealth fifths order upazilas by their Relative Wealth Index and each
hold a fifth of the population. All changes are model counterfactuals at the main parameters.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoai.econ import weighted_groups  # noqa: E402

T = ROOT / "outputs" / "tables"
d = pd.read_csv(T / "qse_padma_main.csv")
d["income_sr_pct"] = 100 * (d.V_hat_sr - 1)
d["population_lr_pct"] = 100 * (d.L_hat_lr - 1)
d["treated"] = d.D_nat > 0.1
d["fifth"] = weighted_groups(d.rwi.values, d["pop"].values, k=5)
wmean = lambda g, c: float(np.average(g[c], weights=g["pop"]))  # noqa: E731
total_gain = float((d.income_sr_pct * d["pop"]).sum())

out = {"n_upazilas": int(len(d)), "national_real_income_sr_pct": wmean(d, "income_sr_pct"), "wealth_fifths": [],
       "treated": {}, "losers": {}}
for k, g in d.groupby("fifth"):
    out["wealth_fifths"].append({
        "fifth": int(k), "upazilas": int(len(g)), "population_share": float(g["pop"].sum() / d["pop"].sum()),
        "mean_rwi": wmean(g, "rwi"), "real_income_sr_pct": wmean(g, "income_sr_pct"),
        "population_lr_pct": wmean(g, "population_lr_pct"),
        "population_share_treated": float(g.loc[g.treated, "pop"].sum() / g["pop"].sum()),
        "population_share_in_losing_upazilas": float(g.loc[g.income_sr_pct < 0, "pop"].sum() / g["pop"].sum())})
t, r = d[d.treated], d[~d.treated]
out["treated"] = {"upazilas": int(len(t)), "population_share": float(t["pop"].sum() / d["pop"].sum()),
                  "share_of_total_income_gain": float((t.income_sr_pct * t["pop"]).sum() / total_gain),
                  "mean_rwi_treated": wmean(t, "rwi"), "mean_rwi_rest": wmean(r, "rwi")}
lo = d[d.income_sr_pct < 0]
out["losers"] = {"upazilas": int(len(lo)), "population_share": float(lo["pop"].sum() / d["pop"].sum()),
                 "mean_real_income_sr_pct": wmean(lo, "income_sr_pct"), "largest_loss_pct": float(lo.income_sr_pct.min()),
                 "upazilas_by_division": {k: int(v) for k, v in lo.groupby("NAME_1").size().items()}}
(T / "qse_incidence.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
