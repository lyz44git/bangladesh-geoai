"""Does the patch-level split of the U-Net leak across the spatial folds?

The U-Net is trained on 64-pixel patches, each assigned to the fold that covers most of its pixels
(`models.extract_patches`), whereas accuracy is scored on every pixel of fold 0 (`02_builtup.py::evaluate`).
A patch that straddles a block boundary can therefore place fold-0 pixels inside a training or validation patch.
This script counts those pixels and scores the U-Net again without them.

  python scripts/11_patch_fold_check.py     # needs data/composites, data/labels and data/models/unet_cities.pt
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoai import builtup, models  # noqa: E402

_spec = importlib.util.spec_from_file_location("builtup_stage", ROOT / "scripts" / "02_builtup.py")
stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage)


def patch_cover(Z, frac, folds, size=64, stride=64):
    """Pixels inside a kept training (fold >= 2) or validation (fold 1) patch, following `models.extract_patches`."""
    train, val = np.zeros(folds.shape, bool), np.zeros(folds.shape, bool)
    _, h, w = Z.shape
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            win = (slice(r, r + size), slice(c, c + size))
            if np.isnan(frac[win]).mean() > 0.2 or (Z[(slice(None), *win)] == 0).all(0).mean() > 0.2:
                continue
            fold = np.bincount(folds[win].ravel()).argmax()
            if fold >= 2:
                train[win] = True
            elif fold == 1:
                val[win] = True
    return train, val


def main():
    names = [f"city_{n}" for n in stage.CFG["cities"]]
    _, (net, stats) = stage.load_city_model()
    out, pool = {}, {"all": ([], []), "clean": ([], [])}
    n_train = n_val = n_test = 0
    for k, name in enumerate(names):
        train = val = None
        for year in stage.UNET_TRAIN_YEARS:  # the patch grid is the same in both years; the kept patches can differ
            path = str(stage.COMP / f"{name}_{year}.tif")
            g = builtup.grid_of(path)
            window = builtup.Window(0, 0, g.width, g.height)
            F, _, _ = builtup.window_features(path, window, (slice(0, g.height), slice(0, g.width)))
            Z, _ = models.normalise(F[:stage.UNET_CH], stats)
            frac = builtup.read_window(str(stage.LAB / f"{name}_wc{year}.tif"), window)
            folds = models.block_folds(g.height, g.width, g.res, seed=k)
            t, v = patch_cover(Z, frac, folds)
            train, val = (t, v) if train is None else (train | t, val | v)
        p, frac, g = stage.full_maps(name, stage.TRAIN_YEAR, "unet", (net, stats))
        test = models.block_folds(g.height, g.width, g.res, seed=k) == 0
        clean = test & ~train & ~val
        row = dict(test_pixels=int(test.sum()), share_in_training_patch=float(train[test].mean()),
                   share_in_validation_patch=float((val & ~train)[test].mean()))
        for tag, m in (("all", test), ("clean", clean)):
            s = models.binary_metrics(p[m], frac[m])
            row[f"{tag}_test_pixels"] = {k_: s[k_] for k_ in ("n", "precision", "recall", "f1")}
            pool[tag][0].append(p[m]), pool[tag][1].append(frac[m])
        out[name] = row
        n_train += int(train[test].sum()); n_val += int((val & ~train)[test].sum()); n_test += int(test.sum())
        print(name, json.dumps(row))
    row = dict(test_pixels=n_test, share_in_training_patch=n_train / n_test, share_in_validation_patch=n_val / n_test)
    for tag, (ps, ts) in pool.items():
        s = models.binary_metrics(np.concatenate(ps), np.concatenate(ts))
        row[f"{tag}_test_pixels"] = {k_: s[k_] for k_ in ("n", "precision", "recall", "f1")}
    out["pooled"] = row
    print("pooled", json.dumps(row))
    note = ("U-Net of the re-run described in the README; 'clean' excludes fold-0 pixels that lie inside a training "
            "or validation patch in either training year")
    (stage.TAB / "builtup_patch_fold_check.json").write_text(json.dumps({"note": note, "results": out}, indent=2))


if __name__ == "__main__":
    main()
