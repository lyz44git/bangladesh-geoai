"""Built-up classifiers: a pixel-wise gradient-boosting model and a small U-Net.

Both are trained against ESA WorldCover built-up fractions and evaluated on
spatially held-out blocks, so the reported accuracy is not inflated by
neighbouring (autocorrelated) pixels leaking between train and test.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- splits
def block_folds(height: int, width: int, res: float, block_m: float = 4000.0, k: int = 5, seed: int = 0,
                offset=(0, 0)) -> np.ndarray:
    """Assign every pixel to one of k folds by square spatial blocks.

    `offset` (row, col) in pixels lets callers keep folds consistent when a
    big grid is processed in chunks.
    """
    bp = max(1, int(round(block_m / res)))
    r = (np.arange(height) + offset[0]) // bp
    c = (np.arange(width) + offset[1]) // bp
    bid = r[:, None] * 100_003 + c[None, :]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(1 << 16)
    return perm[(bid * 2654435761 % (1 << 16)).astype(np.int64)] % k


def sample_training(F: np.ndarray, frac: np.ndarray, folds: np.ndarray, n_per_class: int, rng,
                    pos_thr: float = 0.5, neg_thr: float = 0.1):
    """Draw balanced built / non-built pixels. Mixed pixels are left out of training."""
    ok = np.isfinite(F).all(0) & np.isfinite(frac)
    out = []
    for lab, cond in ((1, frac >= pos_thr), (0, frac <= neg_thr)):
        idx = np.flatnonzero(ok & cond)
        if idx.size == 0:
            continue
        idx = rng.choice(idx, size=min(n_per_class, idx.size), replace=False)
        out.append((idx, lab))
    if not out:
        return np.empty((0, F.shape[0]), np.float32), np.empty(0, int), np.empty(0, int)
    idx = np.concatenate([i for i, _ in out])
    y = np.concatenate([np.full(i.size, lab) for i, lab in out])
    X = F.reshape(F.shape[0], -1)[:, idx].T
    return X.astype(np.float32), y, folds.reshape(-1)[idx]


# ------------------------------------------------------------------- GBM model
def train_gbm(X, y, Xval=None, yval=None, seed: int = 0):
    import lightgbm as lgb

    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=600, random_state=seed, verbose=-1, n_jobs=4)
    m = lgb.LGBMClassifier(**params)
    if Xval is not None:
        m.fit(X, y, eval_set=[(Xval, yval)], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m.fit(X, y)
    return m


def predict_map(model, F: np.ndarray, batch: int = 2_000_000) -> np.ndarray:
    """Built-up probability for every pixel of a feature stack (NaN where features missing)."""
    nf, h, w = F.shape
    X = F.reshape(nf, -1).T
    ok = np.isfinite(X).all(1)
    p = np.full(h * w, np.nan, np.float32)
    idx = np.flatnonzero(ok)
    for s in range(0, idx.size, batch):
        j = idx[s:s + batch]
        p[j] = model.predict_proba(X[j])[:, 1]
    return p.reshape(h, w)


# ---------------------------------------------------------------------- U-Net
def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_unet(in_ch: int, base: int = 24):
    torch, nn = _torch()

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = block(in_ch, base), block(base, base * 2), block(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.d2 = block(base * 4, base * 2)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.d1 = block(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.head(d1)

    return UNet()


def normalise(F: np.ndarray, stats=None):
    """Standardise channels; NaN -> 0 after scaling. Returns (array, stats)."""
    if stats is None:
        flat = F.reshape(F.shape[0], -1)
        mu = np.nanmean(flat, 1)
        sd = np.nanstd(flat, 1) + 1e-6
        stats = (mu.astype(np.float32), sd.astype(np.float32))
    mu, sd = stats
    Z = (F - mu[:, None, None]) / sd[:, None, None]
    return np.nan_to_num(Z, nan=0.0).astype(np.float32), stats


def extract_patches(Z: np.ndarray, frac: np.ndarray, folds: np.ndarray, size: int = 64, stride: int = 48):
    """Cut (C, size, size) patches with their soft labels and majority fold."""
    xs, ys, fs = [], [], []
    _, h, w = Z.shape
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            lab = frac[r:r + size, c:c + size]
            if np.isnan(lab).mean() > 0.2 or (Z[:, r:r + size, c:c + size] == 0).all(0).mean() > 0.2:
                continue
            xs.append(Z[:, r:r + size, c:c + size])
            ys.append(np.nan_to_num(lab, nan=0.0))
            fs.append(np.bincount(folds[r:r + size, c:c + size].ravel()).argmax())
    return np.stack(xs), np.stack(ys).astype(np.float32), np.array(fs)


def train_unet(Xtr, ytr, Xva, yva, epochs: int = 20, lr: float = 2e-3, batch: int = 32, seed: int = 0, log=print,
               radiometric: float = 0.1):
    """Train with soft-label BCE (WorldCover fraction as target). CPU-friendly.

    `radiometric` adds random per-channel gain/offset noise so the network does
    not key on season- or atmosphere-specific absolute reflectance levels,
    which differ between annual composites.
    """
    torch, nn = _torch()
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = build_unet(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(Xtr) / batch)))
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)[:, None]
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)[:, None]
    best, best_state = np.inf, None
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tot = 0.0
        for s in range(0, len(perm), batch):
            j = perm[s:s + batch]
            xb, yb = Xtr_t[j], ytr_t[j]
            if rng.random() < 0.5:  # flips / rotations: built-up has no preferred orientation
                xb, yb = xb.flip(-1), yb.flip(-1)
            k = int(rng.integers(4))
            xb, yb = torch.rot90(xb, k, (-2, -1)), torch.rot90(yb, k, (-2, -1))
            if radiometric:  # per-sample, per-channel gain/offset jitter (standardised units)
                g = 1 + radiometric * torch.randn(xb.shape[0], xb.shape[1], 1, 1)
                o = radiometric * torch.randn(xb.shape[0], xb.shape[1], 1, 1)
                xb = xb * g + o
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(model(Xva_t[s:s + 64]), yva_t[s:s + 64]).item() for s in range(0, len(Xva_t), 64)]))
        log(f"epoch {ep + 1}/{epochs} train {tot / len(perm):.4f} val {vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_unet(model, Z: np.ndarray, tile: int = 256, pad: int = 16) -> np.ndarray:
    """Sliding-window inference with overlap padding; returns probabilities."""
    torch, _ = _torch()
    c, h, w = Z.shape
    out = np.zeros((h, w), np.float32)
    Zp = np.pad(Z, ((0, 0), (pad, pad + tile), (pad, pad + tile)), mode="reflect")
    with torch.no_grad():
        for r in range(0, h, tile):
            for col in range(0, w, tile):
                x = Zp[:, r:r + tile + 2 * pad, col:col + tile + 2 * pad]
                ph, pw = (-x.shape[1]) % 4, (-x.shape[2]) % 4
                x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
                p = torch.sigmoid(model(torch.from_numpy(x[None]))).numpy()[0, 0]
                hh, ww = min(tile, h - r), min(tile, w - col)
                out[r:r + hh, col:col + ww] = p[pad:pad + hh, pad:pad + ww]
    return out


# -------------------------------------------------------------------- metrics
def binary_metrics(prob: np.ndarray, frac: np.ndarray, thr: float = 0.5, label_thr: float = 0.5) -> dict:
    ok = np.isfinite(prob) & np.isfinite(frac)
    p, t = prob[ok] >= thr, frac[ok] >= label_thr
    tp, fp, fn, tn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(n=int(ok.sum()), precision=float(prec), recall=float(rec), f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
                iou=float(tp / max(tp + fp + fn, 1)), accuracy=float((tp + tn) / max(ok.sum(), 1)),
                built_share_pred=float(p.mean()), built_share_ref=float(t.mean()),
                frac_mae=float(np.mean(np.abs(prob[ok] - frac[ok]))))
