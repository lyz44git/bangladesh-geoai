"""Power of the difference-in-differences design against the effect implied by the spatial model.

Reads committed tables only (no data needed) and writes outputs/tables/padma_power.json:
power of a two-sided 5% test (normal approximation), the standard error and the minimum detectable
effect at 80% power, the upper confidence bound relative to the model-implied effect, and the
model-implied effect in a typical treated cell expressed in 30 m pixels.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoai.econ import minimum_detectable_effect, power  # noqa: E402

T = ROOT / "outputs" / "tables"
CELL_PIXELS = 40 * 40  # the outcome is the built-up share of a 1.2 km cell made of 30 m pixels

did = pd.read_csv(T / "padma_did_results.csv").set_index("model")
se = {"upazila clusters": float(did.loc[[i for i in did.index if i.startswith("main: (2)")][0], "se"]),
      "district clusters": float(did.loc["cluster by district", "se"])}
effects = {"main parameters": json.loads((T / "qse_calibration.json").read_text())["main"]["implied_rf_coef"],
           "mean of 600 prior draws": json.loads((T / "qse_padma_bayes.json").read_text())["implied_rf_coef"]["prior_mean"]}
out = {"standard_errors": se, "model_implied_effects_full_adjustment": effects, "power": {}, "se_needed_for_80pct_power": {}}
for ename, b in effects.items():
    out["se_needed_for_80pct_power"][ename] = b / minimum_detectable_effect(1.0)
    for adj, share in (("full adjustment", 1.0), ("half adjustment", 0.5)):
        out["power"][f"{ename}, {adj}"] = {k: power(b * share, s) for k, s in se.items()}

# Minimum detectable effect of the design, and how it compares with what the model predicts.
b = effects["main parameters"]
out["minimum_detectable_effect_80pct_power"] = {k: minimum_detectable_effect(s) for k, s in se.items()}
out["mde_relative_to_model_effect"] = {k: minimum_detectable_effect(s) / b for k, s in se.items()}

# Upper end of the 95% confidence interval: the largest effect the estimate does not reject.
ci_hi = {"upazila clusters": float(did.loc[[i for i in did.index if i.startswith("main: (2)")][0], "ci_hi"]),
         "district clusters": float(did.loc["cluster by district", "ci_hi"])}
out["upper_confidence_bound"] = ci_hi
out["upper_confidence_bound_relative_to_model_effect"] = {k: v / b for k, v in ci_hi.items()}

# The model-implied effect in a typical treated cell (mean treatment intensity among cells with D > 0.05).
smp = json.loads((T / "padma_sample_description.json").read_text())
effect_pp = b * smp["D_mean_treated"]
out["typical_treated_cell"] = {
    "treatment_intensity_log_points": smp["D_mean_treated"],
    "pixels_per_cell": CELL_PIXELS,
    "baseline_built_pp": smp["built_pre_mean_treated"],
    "baseline_built_pixels": smp["built_pre_mean_treated"] / 100 * CELL_PIXELS,
    "model_implied_effect_pp": effect_pp,
    "model_implied_effect_pixels": effect_pp / 100 * CELL_PIXELS,
    "model_implied_effect_relative_to_baseline_pct": 100 * effect_pp / smp["built_pre_mean_treated"],
}
(T / "padma_power.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
