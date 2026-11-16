"""Power of the difference-in-differences design against the effect implied by the spatial model.

Reads committed tables only (no data needed) and writes outputs/tables/padma_power.json.
Two-sided 5% test, normal approximation: power = Phi(-1.96 + b/se) + Phi(-1.96 - b/se).
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import norm

T = Path(__file__).resolve().parents[1] / "outputs" / "tables"


def power(effect, se, z=1.959964):
    return float(norm.cdf(-z + effect / se) + norm.cdf(-z - effect / se))


did = pd.read_csv(T / "padma_did_results.csv").set_index("model")
se = {"upazila clusters": float(did.loc[[i for i in did.index if i.startswith("main: (2)")][0], "se"]),
      "district clusters": float(did.loc["cluster by district", "se"])}
effects = {"main parameters": json.loads((T / "qse_calibration.json").read_text())["main"]["implied_rf_coef"],
           "mean of 600 prior draws": json.loads((T / "qse_padma_bayes.json").read_text())["implied_rf_coef"]["prior_mean"]}
out = {"standard_errors": se, "model_implied_effects_full_adjustment": effects, "power": {}, "se_needed_for_80pct_power": {}}
for ename, b in effects.items():
    out["se_needed_for_80pct_power"][ename] = b / (1.959964 + 0.841621)
    for adj, share in (("full adjustment", 1.0), ("half adjustment", 0.5)):
        out["power"][f"{ename}, {adj}"] = {k: power(b * share, s) for k, s in se.items()}
(T / "padma_power.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
