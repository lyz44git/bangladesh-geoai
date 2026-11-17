"""Built-up mapping: labels -> train/evaluate (GBM vs U-Net) -> annual predictions.

  python scripts/02_builtup.py labels
  python scripts/02_builtup.py cities     # train + compare models on the six cities (20 m), predict 2018-2026
  python scripts/02_builtup.py padma      # train on the Padma region (30 m), predict 2018-2026

Labels are ESA WorldCover 2021 built-up fractions; the 2021 composite (dry
season Nov 2020 - Mar 2021) is the training image. Accuracy is measured on
spatial blocks (4 km) never seen in training.
"""
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoai import builtup, models  # noqa: E402
from geoai.grid import grid_around, grid_from_lonlat  # noqa: E402

CFG = json.loads((ROOT / "configs/aois.json").read_text())
COMP, LAB, PRED, MOD = (ROOT / "data" / d for d in ("composites", "labels", "builtup", "models"))
TAB = ROOT / "outputs/tables"
for d in (LAB, PRED, MOD, TAB):
    d.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
for _n in ("rasterio", "rasterio._env", "rasterio.env", "rasterio.session"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("builtup").info
TRAIN_YEAR = 2021
UNET_CH = 12  # 6 bands + 6 indices; the CNN learns its own texture


def city_grid(name):
    c = CFG["cities"][name]
    return grid_around(c["lon"], c["lat"], c["half_km"], CFG["city_res_m"])


def padma_grid():
    p = CFG["padma_region"]
    return grid_from_lonlat(p["lon_min"], p["lat_min"], p["lon_max"], p["lat_max"], CFG["padma_res_m"])


def aois():
    out = {f"city_{n}": city_grid(n) for n in CFG["cities"]}
    out["padma"] = padma_grid()
    return out


def stage_labels():
    for name, g in aois().items():
        for y in (2021, 2020):
            t = time.time()
            builtup.write_label_raster(g, str(LAB / f"{name}_wc{y}.tif"), year=y)
            log(f"labels {name} {y} {time.time() - t:.0f}s")


def collect(names, n_per_window=4000, seed=0):
    """Balanced training pixels from each AOI's 2021 composite, with spatial folds."""
    rng = np.random.default_rng(seed)
    Xs, ys, fs, src = [], [], [], []
    for k, name in enumerate(names):
        path = str(COMP / f"{name}_{TRAIN_YEAR}.tif")
        g = builtup.grid_of(path)
        for core, read, sl in builtup.windows(g.width, g.height):
            F, fnames, _ = builtup.window_features(path, read, sl)
            frac = builtup.read_window(str(LAB / f"{name}_wc{TRAIN_YEAR}.tif"), core)
            folds = models.block_folds(core.height, core.width, g.res, offset=(core.row_off, core.col_off), seed=k)
            X, y, f = models.sample_training(F, frac, folds, n_per_window, rng)
            Xs.append(X), ys.append(y), fs.append(f), src.append(np.full(len(y), k))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs), np.concatenate(src), fnames


def evaluate(names, prob_fn, year=TRAIN_YEAR, label_year=TRAIN_YEAR, test_fold=0):
    """Metrics on held-out fold pixels (all pixels, including mixed ones), per AOI and pooled."""
    rows, pool_p, pool_t = [], [], []
    for k, name in enumerate(names):
        p, frac, g = prob_fn(name, year)
        folds = models.block_folds(g.height, g.width, g.res, seed=k)
        m = folds == test_fold
        rows.append(dict(aoi=name, **models.binary_metrics(p[m], frac[m])))
        pool_p.append(p[m]), pool_t.append(frac[m])
    rows.append(dict(aoi="pooled", **models.binary_metrics(np.concatenate(pool_p), np.concatenate(pool_t))))
    return pd.DataFrame(rows)


def full_maps(name, year, model_kind, model):
    """Full-grid probability + label for one AOI (cities are small enough to hold in memory)."""
    path = str(COMP / f"{name}_{year}.tif")
    g = builtup.grid_of(path)
    frac = builtup.read_window(str(LAB / f"{name}_wc{min(max(year, 2020), 2021)}.tif"),
                               builtup.Window(0, 0, g.width, g.height))
    if model_kind == "gbm":
        p = np.full((g.height, g.width), np.nan, np.float32)
        for core, read, sl in builtup.windows(g.width, g.height):
            F, _, _ = builtup.window_features(path, read, sl)
            p[core.row_off:core.row_off + core.height, core.col_off:core.col_off + core.width] = models.predict_map(model, F)
    else:
        net, stats = model
        F, _, _ = builtup.window_features(path, builtup.Window(0, 0, g.width, g.height),
                                          (slice(0, g.height), slice(0, g.width)))
        Z, _ = models.normalise(F[:UNET_CH], stats)
        p = models.predict_unet(net, Z)
        p[~np.isfinite(F[:6]).all(0)] = np.nan
    return p, frac, g


def train_gbm(names, tag):
    X, y, f, src, fnames = collect(names)
    log(f"{tag}: {len(y)} samples ({y.mean():.2f} built), {X.shape[1]} features")
    tr, va = f >= 2, f == 1
    gbm = models.train_gbm(X[tr], y[tr], X[va], y[va])
    imp = pd.DataFrame({"feature": fnames, "gain": gbm.booster_.feature_importance("gain")})
    imp.sort_values("gain", ascending=False).to_csv(TAB / f"builtup_gbm_importance_{tag}.csv", index=False)
    with open(MOD / f"gbm_{tag}.pkl", "wb") as fh:
        pickle.dump(gbm, fh)
    return gbm


UNET_TRAIN_YEARS = (2020, 2021)  # two seasons, each with its own WorldCover map


def train_unet(names):
    rng = np.random.default_rng(1)
    Xs, ys, fs, samples = [], [], [], []
    full = lambda g: (builtup.Window(0, 0, g.width, g.height), (slice(0, g.height), slice(0, g.width)))  # noqa: E731
    for k, name in enumerate(names):  # pooled normalisation statistics from a pixel sample
        path = str(COMP / f"{name}_{TRAIN_YEAR}.tif")
        g = builtup.grid_of(path)
        F, _, _ = builtup.window_features(path, *full(g))
        flat = F[:UNET_CH].reshape(UNET_CH, -1)
        samples.append(flat[:, rng.choice(flat.shape[1], 50_000, replace=False)])
    S = np.concatenate(samples, 1)
    stats = (np.nanmean(S, 1).astype(np.float32), (np.nanstd(S, 1) + 1e-6).astype(np.float32))
    for year in UNET_TRAIN_YEARS:
        for k, name in enumerate(names):
            path = str(COMP / f"{name}_{year}.tif")
            g = builtup.grid_of(path)
            F, _, _ = builtup.window_features(path, *full(g))
            Z, _ = models.normalise(F[:UNET_CH], stats)
            frac = builtup.read_window(str(LAB / f"{name}_wc{year}.tif"), full(g)[0])
            folds = models.block_folds(g.height, g.width, g.res, seed=k)  # same blocks in both years
            X, y, fo = models.extract_patches(Z, frac, folds, size=64, stride=64)
            Xs.append(X), ys.append(y), fs.append(fo)
    X, y, fo = np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs)
    log(f"unet patches: {len(X)} (train {np.sum(fo >= 2)}, val {np.sum(fo == 1)})")
    net = models.train_unet(X[fo >= 2], y[fo >= 2], X[fo == 1], y[fo == 1], epochs=18, log=log)
    import torch

    torch.save({"state": net.state_dict(), "stats": stats, "in_ch": UNET_CH}, MOD / "unet_cities.pt")
    return net, stats


def stage_cities():
    names = [f"city_{n}" for n in CFG["cities"]]
    gbm = train_gbm(names, "cities")
    res = {"gbm": evaluate(names, lambda n, y: full_maps(n, y, "gbm", gbm))}
    unet = train_unet(names)
    res["unet"] = evaluate(names, lambda n, y: full_maps(n, y, "unet", unet))
    # The 2020 composite against WorldCover 2020 (v100). Out-of-time for the GBM, which is trained on 2021 only; the
    # U-Net is trained on both seasons (UNET_TRAIN_YEARS), so for it this is a second in-time score on held-out blocks.
    res["gbm_2020"] = evaluate(names, lambda n, y: full_maps(n, 2020, "gbm", gbm))
    res["unet_2020"] = evaluate(names, lambda n, y: full_maps(n, 2020, "unet", unet))
    tab = pd.concat([d.assign(model=k) for k, d in res.items()])
    tab.to_csv(TAB / "builtup_model_comparison_cities.csv", index=False)
    pooled = tab[tab.aoi == "pooled"].set_index("model")
    log("\n" + pooled[["f1", "iou", "precision", "recall", "frac_mae"]].round(3).to_string())
    best = "unet" if pooled.loc["unet", "f1"] > pooled.loc["gbm", "f1"] else "gbm"
    (MOD / "best_cities.txt").write_text(best)
    log(f"cities trained; best={best}")
    stage_predict_cities()


def load_city_model():
    best = (MOD / "best_cities.txt").read_text().strip()
    if best == "gbm":
        with open(MOD / "gbm_cities.pkl", "rb") as fh:
            return best, pickle.load(fh)
    import torch

    ck = torch.load(MOD / "unet_cities.pt", weights_only=False)
    net = models.build_unet(ck["in_ch"])
    net.load_state_dict(ck["state"])
    net.eval()
    return best, (net, ck["stats"])


def stage_predict_cities():
    """Predict every available city-year composite with the selected model (skips existing outputs)."""
    best, model = load_city_model()
    for name in [f"city_{n}" for n in CFG["cities"]]:
        for y in CFG["years"]:
            out = PRED / f"{name}_{y}.tif"
            if out.exists() or not (COMP / f"{name}_{y}.tif").exists():
                continue
            p, _, g = full_maps(name, y, best, model)
            write_prob(out, p, g)
            log(f"predicted {name} {y} ({best})")


def write_prob(out, p, g):
    import rasterio

    prof = dict(driver="GTiff", width=g.width, height=g.height, count=1, dtype="uint8", crs=g.crs,
                transform=g.transform, nodata=255, compress="deflate")
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(np.where(np.isfinite(p), np.round(p * 250), 255).astype(np.uint8), 1)


def best_threshold(p, frac, label_thr=0.5):
    """Probability cut-off maximising F1 on validation pixels."""
    grid = np.linspace(0.3, 0.95, 27)
    f1 = [models.binary_metrics(p, frac, thr=t, label_thr=label_thr)["f1"] for t in grid]
    return float(grid[int(np.argmax(f1))])


def stage_calibrate_gbm():
    """GBM with an F1-tuned threshold (validation fold 1), evaluated on test fold 0.

    The GBM is trained on class-balanced samples, so its 0.5 cut-off over-predicts
    built-up; tuning the threshold on validation pixels gives it a fair comparison.
    """
    names = [f"city_{n}" for n in CFG["cities"]]
    with open(MOD / "gbm_cities.pkl", "rb") as fh:
        gbm = pickle.load(fh)
    maps = {n: full_maps(n, TRAIN_YEAR, "gbm", gbm) for n in names}
    vp, vt = [], []
    for k, n in enumerate(names):
        p, frac, g = maps[n]
        m = models.block_folds(g.height, g.width, g.res, seed=k) == 1
        vp.append(p[m]), vt.append(frac[m])
    thr = best_threshold(np.concatenate(vp), np.concatenate(vt))
    rows, pp, tt = [], [], []
    for k, n in enumerate(names):
        p, frac, g = maps[n]
        m = models.block_folds(g.height, g.width, g.res, seed=k) == 0
        rows.append(dict(aoi=n, **models.binary_metrics(p[m], frac[m], thr=thr)))
        pp.append(p[m]), tt.append(frac[m])
    rows.append(dict(aoi="pooled", **models.binary_metrics(np.concatenate(pp), np.concatenate(tt), thr=thr)))
    tab = pd.read_csv(TAB / "builtup_model_comparison_cities.csv")
    tab = tab[tab.model != "gbm_tuned"]
    new = pd.DataFrame(rows).assign(model="gbm_tuned", threshold=thr)
    pd.concat([tab, new]).to_csv(TAB / "builtup_model_comparison_cities.csv", index=False)
    log(f"gbm tuned threshold {thr:.2f}: pooled F1 {rows[-1]['f1']:.3f}")


def unet_window_predict(net, stats, path, size=1024, halo=32):
    """U-Net probabilities over a large composite, window by window with a context halo."""
    g = builtup.grid_of(path)
    out = np.full((g.height, g.width), np.nan, np.float32)
    for core, read, sl in builtup.windows(g.width, g.height, size=size, halo=halo):
        refl, _ = builtup.read_composite(path, window=read)
        F, _ = builtup.feature_stack(refl, texture_sizes=())
        Z, _ = models.normalise(F[:UNET_CH], stats)
        p = models.predict_unet(net, Z)
        p[~np.isfinite(refl).all(0)] = np.nan
        out[core.row_off:core.row_off + core.height, core.col_off:core.col_off + core.width] = p[sl]
    return out, g


def stage_padma(max_patches=5000, per_window=60):
    """Padma region (30 m): U-Net (main) and threshold-tuned GBM (comparison), then annual predictions."""
    import torch

    path = str(COMP / f"padma_{TRAIN_YEAR}.tif")
    lab = str(LAB / f"padma_wc{TRAIN_YEAR}.tif")
    g = builtup.grid_of(path)
    ck_path = MOD / "unet_padma.pt"
    rng = np.random.default_rng(2)
    if ck_path.exists():
        ck = torch.load(ck_path, weights_only=False)
        net = models.build_unet(ck["in_ch"])
        net.load_state_dict(ck["state"])
        net.eval()
        stats = ck["stats"]
    else:
        wins = list(builtup.windows(g.width, g.height, size=1024, halo=0))
        samples = []
        for core, read, sl in wins:
            refl, _ = builtup.read_composite(path, window=read)
            F, _ = builtup.feature_stack(refl, texture_sizes=())
            flat = F[:UNET_CH].reshape(UNET_CH, -1)
            ok = np.isfinite(flat).all(0)
            if ok.sum() > 1000:
                samples.append(flat[:, rng.choice(np.flatnonzero(ok), 2000, replace=False)])
        S = np.concatenate(samples, 1)
        stats = (np.nanmean(S, 1).astype(np.float32), (np.nanstd(S, 1) + 1e-6).astype(np.float32))
        Xs, ys, fs = [], [], []
        for year in UNET_TRAIN_YEARS:
            ypath, ylab = str(COMP / f"padma_{year}.tif"), str(LAB / f"padma_wc{year}.tif")
            for core, read, sl in wins:
                refl, _ = builtup.read_composite(ypath, window=read)
                if np.isnan(refl[0]).mean() > 0.5:
                    continue
                F, _ = builtup.feature_stack(refl, texture_sizes=())
                Z, _ = models.normalise(F[:UNET_CH], stats)
                frac = builtup.read_window(ylab, core)
                folds = models.block_folds(core.height, core.width, g.res, offset=(core.row_off, core.col_off), seed=0)
                try:
                    X, y, fo = models.extract_patches(Z, frac, folds, size=64, stride=64)
                except ValueError:  # no usable patch in this window
                    continue
                if len(X) > per_window:  # bound memory: only max_patches are used in training anyway
                    k = rng.choice(len(X), per_window, replace=False)
                    X, y, fo = X[k], y[k], fo[k]
                Xs.append(X), ys.append(y), fs.append(fo)
        X, y, fo = np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs)
        tr = np.flatnonzero(fo >= 2)
        va = np.flatnonzero(fo == 1)
        tr = rng.choice(tr, min(max_patches, tr.size), replace=False)
        va = rng.choice(va, min(max_patches // 3, va.size), replace=False)
        log(f"padma unet patches: train {tr.size}, val {va.size} (of {len(X)})")
        net = models.train_unet(X[tr], y[tr], X[va], y[va], epochs=14, log=log)
        torch.save({"state": net.state_dict(), "stats": stats, "in_ch": UNET_CH}, ck_path)
        del X, y
    gbm = train_gbm(["padma"], "padma")
    # Held-out evaluation on fold-0 blocks (and threshold tuning for GBM on fold 1).
    p_unet, _ = unet_window_predict(net, stats, path)
    frac = builtup.read_window(lab, builtup.Window(0, 0, g.width, g.height))
    folds = models.block_folds(g.height, g.width, g.res, seed=0)
    p_gbm = np.full((g.height, g.width), np.nan, np.float32)
    for core, read, sl in builtup.windows(g.width, g.height):
        F, _, _ = builtup.window_features(path, read, sl)
        p_gbm[core.row_off:core.row_off + core.height, core.col_off:core.col_off + core.width] = models.predict_map(gbm, F)
    thr = best_threshold(p_gbm[folds == 1], frac[folds == 1])
    te = folds == 0
    met = pd.DataFrame([
        dict(aoi="padma", model="unet", threshold=0.5, **models.binary_metrics(p_unet[te], frac[te])),
        dict(aoi="padma", model="gbm", threshold=0.5, **models.binary_metrics(p_gbm[te], frac[te])),
        dict(aoi="padma", model="gbm_tuned", threshold=thr, **models.binary_metrics(p_gbm[te], frac[te], thr=thr)),
    ])
    met.to_csv(TAB / "builtup_model_padma.csv", index=False)
    log("\n" + met.round(3).to_string())
    del p_gbm, frac, folds
    for y in CFG["years"]:
        src = COMP / f"padma_{y}.tif"
        out = PRED / f"padma_{y}.tif"
        if out.exists() or not src.exists():
            continue
        p = p_unet if y == TRAIN_YEAR else unet_window_predict(net, stats, str(src))[0]
        write_prob(out, p, g)
        log(f"predicted padma {y}")


if __name__ == "__main__":
    {"labels": stage_labels, "cities": stage_cities, "predict-cities": stage_predict_cities,
     "calibrate-gbm": stage_calibrate_gbm, "padma": stage_padma}[sys.argv[1]]()
