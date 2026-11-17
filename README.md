# Evaluating a major transport investment with open satellite data: the Padma Bridge, Bangladesh

[![tests](https://github.com/lyz44git/bangladesh-geoai/actions/workflows/tests.yml/badge.svg)](https://github.com/lyz44git/bangladesh-geoai/actions/workflows/tests.yml)

**Summary.** Open satellite data are sufficient for measurement and monitoring, but four years after opening they cannot
yet support an evaluation of the bridge. Free Sentinel-2 imagery measures built-up land accurately enough for city
monitoring (F1 = 0.80) and predicts sub-district wealth accurately enough for geographic targeting (out-of-area
R² = 0.83). Four years after the Padma Bridge opened, the same imagery shows no detectable change in built-up land. A
calibrated spatial equilibrium model accounts for this result: the effect it predicts is one that the evaluation design
has only 8–23% power to detect. The model places the national welfare gain at +0.47%, concentrated in the southwest.

<img src="figures/figure1.png" alt="Four panels following the study's four steps: wealth-prediction accuracy by feature set; built-up growth in six cities; event study of the bridge's effect on built-up land; map of model-implied income change by upazila." width="100%">

**Figure 1 | From measurement to evaluation.** **a**, Measurement: out-of-sample R² for upazila wealth by feature
set under 50 km spatial-block (filled) and random (open) cross-validation. **b**, Description: growth in built-up
area in the six largest cities, 2020–2026, split by where the new building is. **c**, Evaluation: event-study
estimates of the bridge's effect on built-up share across 34,626 cells with 95% intervals (shaded), 2022 as
reference; the negative naive estimates vanish once trends by baseline density are controlled. **d**,
Interpretation: short-run real income change implied by the spatial equilibrium model, by upazila (square: the
bridge); inset, the same change against the reduction in travel time to Dhaka. Drawn by
[`figures/make_figure1.py`](figures/make_figure1.py) from the committed result tables and the GADM 4.1 boundaries,
which are not stored here.

Technical report: [`docs/TECHNICAL_REPORT_EN.md`](docs/TECHNICAL_REPORT_EN.md) · Indicator atlas for 544 upazilas:
[`docs/atlas/ATLAS.md`](docs/atlas/ATLAS.md)

## Research question and design

The Padma Bridge, a 6.15 km road-rail bridge opened on 25 June 2022, replaced ferry crossings that often involved
hours of queuing and connected the southwestern divisions to Dhaka. Ex-ante appraisals projected national GDP gains
above 1%. Household surveys that could test this projection become available only years later, whereas satellite
imagery is free, annual and covers every upazila. The study examines whether such imagery can support an early
evaluation, proceeding in four steps, each a precondition for the next:

| Step | Question | Approach | Answer |
|---|---|---|---|
| 1. Measure | Is imagery accurate enough for annual measures of built-up land and wealth? | U-Net trained on WorldCover labels; wealth prediction under 50 km spatial-block cross-validation | Yes, for city monitoring and upazila-level targeting |
| 2. Describe | What does it show where change is rapid? | Built-up area in the six largest cities, 2020–2026 | Growth of 27–72%, mostly at the urban edge |
| 3. Evaluate | Did the bridge change where new built-up land appears? | Continuous-treatment difference-in-differences and event study, 34,626 cells × 7 seasons | No detectable effect once selection on trends is removed |
| 4. Interpret | Is the null evidence that the bridge had no effect? | Two-sector spatial equilibrium model of 544 upazilas; power of the design against the model's prediction | No: 8–23% power; the model implies a welfare gain of +0.47% |

## 1. Measurement: accuracy is sufficient for monitoring and targeting

| Built-up classifier, six cities pooled (spatially held-out blocks) | Precision | Recall | F1 |
|---|---|---|---|
| U-Net | 0.86 | 0.75 | 0.80 |
| Gradient boosting, default threshold | 0.52 | 0.96 | 0.67 |

| Wealth prediction (Meta Relative Wealth Index) | R², grid cells | R², upazila means |
|---|---|---|
| Spatial blocks (50 km) | 0.45 | 0.83 |
| Random folds | 0.50 | 0.92 |

- **Compositing, more than the classifier, determines the stability of change over time.** Relative to least-cloudy
  compositing, the
  phenology-balanced design lowers the year-to-year reversal rate by 25–52% and the trend-residual standard deviation
  by 29–75% across the six cities. Without it, seasonal variation would register as urban change and would
  contaminate the event study in step 3.
- **Accuracy declines where settlement is sparse.** In the rural Padma region (1.3% built-up) the U-Net reaches F1 = 0.66
  (precision 0.79, recall 0.56).
- **Random cross-validation overstates wealth accuracy**; the spatial-block figures are the relevant measure for
  out-of-area prediction. In the best imagery model, built-up land cover alone accounts for 31% of the gain.

The wealth index also sets relative wages in the spatial model of step 4; the imagery model indicates that this
input can be largely reproduced from imagery out of area.

![Wealth predicted from imagery for 26,556 grid cells out of area; upazila means of prediction against label; accuracy by feature set under spatial-block and random cross-validation; year-to-year noise of built-up maps under two compositing designs](outputs/figures/measurement.png)

## 2. Description: urban growth is concentrated at the edge

| City | Built-up growth, 2020–2026 | Share of new built-up land at the edge |
|---|---|---|
| Dhaka | +27% | 58% |
| Chattogram | +37% | 57% |
| Khulna | +32% | 50% |
| Rajshahi | +48% | 64% |
| Sylhet | +41% | 55% |
| Barishal | +72% | 61% |

Edge expansion accounts for 50–64% of new built-up land, and radial density gradients flatten in every city with a
measurable gradient. In Dhaka and Chattogram the largest additions lie 4–7 km from the center. Built-up area in the six cities grows by 4.0–9.5% a year. By contrast, the rural areas the
bridge reaches start from 0.37% built-up, so any bridge effect on built-up land is necessarily small in absolute terms,
which limits the power of the evaluation (step 4).

![Built-up land in 2020 and land added by 2026 in six cities, each with a 10 km scale bar](outputs/figures/urban_change_maps.png)

![Smoothed built-up area by city; built-up share added between 2020 and 2026 by distance from the city center; density gradients in 2020 and 2026](outputs/figures/urban_area_and_profiles.png)

## 3. Evaluation: no detectable effect of the bridge on built-up land

Treatment is the modeled reduction in travel time to Dhaka (log points), which varies continuously across 34,626
cells. Effect of a one-log-point reduction on built-up share (percentage points):

| Specification | Coefficient | 95% CI | Randomization p |
|---|---|---|---|
| (1) Cell and year fixed effects | −0.339 | −0.58 to −0.10 | 0.008 |
| (2) + year × baseline-built bin, distance, water | 0.060 | −0.03 to 0.15 | 0.92 |
| (3) + division × year fixed effects | −0.002 | −0.13 to 0.12 | 0.98 |
| (2) with district clusters | 0.060 | −0.11 to 0.23 | |

- **The negative naive estimate reflects selection on trends.** Treated areas are far more rural (0.37% built-up before
  opening against 2.22%) and add fewer percentage points each year. Controlling for trends by baseline density removes
  the pre-trends, and the estimate becomes a null.
- **The null is robust** to ferry speed, the outcome definition, and dropping cells near the bridge or near Dhaka. A
  binary-treatment version of (2) is positive (0.097, CI 0.03 to 0.17) but does not survive district clustering or
  randomization inference.

A null result is consistent either with no effect or with a test that lacks the power to detect one; step 4
distinguishes between the two.

![Reduction in travel time to Dhaka on a 500 m grid; event-study estimates under three specifications; the coefficient under alternative choices](outputs/figures/padma_treatment_and_event_study.png)

## 4. Interpretation: the null is uninformative, and the model implies a gain of +0.47%

- **The model's predicted effect lies within the confidence interval.** At full long-run adjustment the model implies
  a built-up coefficient of 0.059, almost exactly the point estimate of 0.060.
- **The design cannot detect an effect of that size.** Its power is 8–10% with district clustering and 17–23% with
  upazila clustering; 80% power would need a standard error about one-fifth of the district-clustered one.
- **Welfare and incidence.** National welfare rises by 0.47% at the main parameters (0.30–0.79% across the parameter
  grid). Real income rises 1.9% in treated upazilas and 0.15% elsewhere; Rajshahi and Rangpur lose slightly through
  trade diversion.
- **The non-traded sector matters.** Omitting it raises the estimated gain to 0.79%, about 70% higher.

![Long-run population change by upazila; welfare gain across parameter values; model-implied coefficient against the estimate; power of the causal design](outputs/figures/qse_padma.png)

## Implications

1. **Targeting.** Satellite wealth maps are fit for allocating resources across upazilas, provided they are validated
   with spatial-block rather than random cross-validation.
2. **Monitoring.** Any satellite outcome used in a panel needs temporal-consistency diagnostics; the reliability of
   year-to-year change depends more on compositing than on the classifier.
3. **Early evaluation.** Three to four years of built-up imagery should not be read as a verdict on a transport
   investment. The expected effect size and the power of the design should be set out first, for example from a
   calibrated model.
4. **Appraisal.** Spatial models used for appraisal should include a non-traded sector and report gains and losses by
   region.

## Limitations

- Wealth labels are model-based gridded layers, not household survey microdata.
- The 2018–2019 Sentinel-2 archive comes from a different processing baseline and cannot extend the pre-period; a
  placebo opening in 2020 fails for that measurement reason.
- Treatment varies smoothly over space, and the Dhaka–Mawa–Bhanga expressway (2020) and the Padma rail link
  (2023–24) cannot be separated from the bridge.
- The spatial model is static, covers goods trade only and is calibrated, not estimated; its welfare numbers exclude
  passenger time savings and commuting.

## Code map

| Step | Module | Code | What it does |
|---|---|---|---|
| 1 | Compositing | `src/geoai/stac.py`, `composite.py` | Phenology-balanced dry-season composites with orbit-aware scene selection and gap filling |
| 1 | Built-up mapping | `builtup.py`, `labels.py`, `scripts/02_builtup.py` | U-Net and gradient-boosting classifiers trained on WorldCover labels, validated on spatially held-out blocks |
| 1 | Wealth mapping | `features.py`, `mosaiks.py`, `models.py`, `scripts/05_wealth.py` | Predicting the Relative Wealth Index from imagery under 50 km spatial-block cross-validation |
| 2 | Urban expansion | `scripts/03_urban_expansion.py` | Growth, infill / edge / outlying decomposition, radial density gradients, temporal-consistency diagnostics |
| 3 | Travel time | `travel.py`, `scripts/04_travel_time.py` | 500 m least-cost travel time to Dhaka with and without the bridge |
| 3 | Causal design | `econ.py`, `scripts/06_padma_causal.py` | Continuous-treatment difference-in-differences and event study, with randomization inference |
| 4 | Spatial model | `qse.py`, `scripts/07_spatial_model.py` | Two-sector quantitative spatial equilibrium for 544 upazilas, solved by exact hat algebra; sensitivity and Bayesian reweighting |
| 4 | Power | `scripts/09_power.py` | Power of the causal design against the model-implied effect |
| | Atlas and figures | `scripts/08_atlas.py`, `figures/` | Upazila indicator atlas; the figures in this README and the report |

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ".[dl]" for the U-Net (PyTorch)

python scripts/00_fetch_overture.py                           # transport network
python scripts/01_build_composites.py cities|padma|national   # Sentinel-2 composites
python scripts/02_builtup.py labels|cities|calibrate-gbm|padma
python scripts/03_urban_expansion.py
python scripts/04_travel_time.py
python scripts/05_wealth.py
python scripts/06_padma_causal.py
python scripts/07_spatial_model.py times                      # travel-time matrices with and without the bridge
python scripts/07_spatial_model.py model                      # calibration, counterfactuals, Bayesian reweighting
python scripts/08_atlas.py
python scripts/09_power.py                                    # reads committed tables only
```

The pipeline runs on a four-CPU machine without a GPU. Raw data and intermediate rasters go to a git-ignored `data/`
folder; result tables and figures are in `outputs/`.

The figures in this README, the report and the atlas are drawn by `python figures/make_figure1.py` and
`python figures/make_report_figures.py`. Both need the GADM level-3 boundary file; the second also reads the pipeline
rasters in `data/` for the wealth, travel-time and city maps, and falls back to upazila-level maps without them.

**Reproduction check.** The pipeline was re-run from the raw data on a separate machine on 16 November 2026 for the six
cities, the national composite and the wealth study (steps 1–2 and the travel-time model; the Padma-region composites
and the causal and model steps were not re-run). Classifier scores reproduced within 0.005 and the headline wealth R²
within 0.003; across all feature sets the largest change was +0.05 (Sentinel-2 statistics alone). City growth
reproduced within 2.0 percentage points. The published tables are unchanged. The re-run also identified two issues, now
fixed or documented:

- `01_build_composites.py` now calls the orbit-aware gap filling (`composite.fill_gaps`) after each composite. Without
  it, eastern Khulna had no clear observation in 2026 and its built-up area was understated by more than half.
- On macOS, LightGBM, PyTorch and scikit-learn can load three different OpenMP runtimes, and U-Net training then hangs
  at the first epoch. Linking all three to one runtime (for example PyTorch's `libomp.dylib`) avoids it; Linux is not
  affected.

## Tests

```bash
pytest -q
```

17 offline unit tests cover the grid, compositing, labels, features, models, the fixed-effects estimators and the
equilibrium solver (one needs PyTorch and is skipped without it). They run on every push.

## Data and licenses

All inputs are openly available. No data are stored in this repository; the scripts stream or download them.

| Source | Use | License |
|---|---|---|
| Sentinel-2 L2A (ESA/Copernicus), via AWS Open Data and the Earth Search STAC API | Annual dry-season composites, 2018–2026 | Copernicus Sentinel data terms (free use with attribution) |
| ESA WorldCover 2020 and 2021, 10 m | Built-up training labels, land-cover shares, water mask | CC BY 4.0 |
| Meta Relative Wealth Index (Chi et al. 2022), via HDX | Wealth-mapping target (26,544 grid points) | CC BY-NC 4.0 |
| Flowminder/WorldPop Bangladesh 1 km poverty map (2011) | Alternative target and external benchmark | CC BY 4.0 |
| Meta/CIESIN High Resolution Settlement Layer | Population by cell and upazila | CC BY 4.0 |
| Overture Maps transportation | Road, rail and ferry network for the travel-time model | ODbL 1.0 |
| GADM 4.1 | Districts and upazilas | Free for academic and non-commercial use; no redistribution |

Attribution: contains modified Copernicus Sentinel data (2018–2026); © ESA WorldCover project / contains modified
Copernicus Sentinel data (2020, 2021) processed by the ESA WorldCover consortium; © OpenStreetMap contributors,
Overture Maps Foundation.

No boundary files or source rasters are redistributed here. Tables, maps and the atlas in `outputs/` and `docs/` are
derived aggregates; those that include the Relative Wealth Index inherit its non-commercial terms, and the maps use
GADM boundaries under its academic-use terms.

**Use.** This is a non-commercial research project. The data and all derived outputs in this repository are used
for research only.

## License

Code is released under the MIT License (see `LICENSE`). The datasets and the derived outputs keep the terms listed
under Data and licenses.

## Notes

Developed with an AI-assisted coding workflow. The research questions, analysis plans and interpretation of results
were led by the author. Three material errors caught by the verification protocol (observation-level binning, an
overview-based population raster and an orbit-coverage gap) are documented in the technical report.
