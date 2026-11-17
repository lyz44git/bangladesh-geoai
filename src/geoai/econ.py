"""Panel regressions with multiple absorbed fixed effects and cluster-robust SEs, and design calculations.

Fixed effects are removed by alternating projections (the same idea as
reghdfe / fixest), then OLS is run on the demeaned data. Absorbed effects
nested within clusters are not counted in the small-sample correction,
matching common practice.

`power` and `minimum_detectable_effect` use the normal approximation for a two-sided test;
`weighted_groups` splits units into groups holding equal shares of a weight (for example population).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def demean(X: np.ndarray, groups: list[np.ndarray], tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    """Residualise columns of X on several sets of categorical fixed effects."""
    X = X.astype(np.float64).copy()
    codes = [pd.factorize(g)[0] for g in groups]
    counts = [np.bincount(c) for c in codes]
    for _ in range(max_iter):
        prev = X.copy()
        for c, n in zip(codes, counts):
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(c, weights=X[:, j]) / n)[c]
        if np.max(np.abs(X - prev)) < tol:
            break
    return X


def ols_cluster(y: np.ndarray, X: np.ndarray, cluster: np.ndarray):
    """OLS with one-way cluster-robust (CR1) covariance. Returns (beta, cov)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    cl = pd.factorize(cluster)[0]
    G = cl.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, cl, X * u[:, None])
    meat = S.T @ S
    n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k)
    return beta, adj * XtX_inv @ meat @ XtX_inv


def feols(df: pd.DataFrame, y: str, x: list[str], fe: list[str], cluster: str):
    """Regress y on x absorbing `fe` (column names; use 'a^b' for interacted FE)."""
    groups = []
    for f in fe:
        if "^" in f:
            a, b = f.split("^")
            groups.append((df[a].astype(str) + "_" + df[b].astype(str)).values)
        else:
            groups.append(df[f].values)
    M = demean(df[[y] + x].values, groups)
    beta, cov = ols_cluster(M[:, 0], M[:, 1:], df[cluster].values)
    se = np.sqrt(np.diag(cov))
    out = pd.DataFrame({"coef": beta, "se": se}, index=x)
    out["t"] = out.coef / out.se
    out["ci_lo"], out["ci_hi"] = out.coef - 1.96 * out.se, out.coef + 1.96 * out.se
    return out, cov


def power(effect: float, se: float, alpha: float = 0.05) -> float:
    """Power of a two-sided test of size alpha against a true effect, given the standard error."""
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(-z + effect / se) + norm.cdf(-z - effect / se))


def minimum_detectable_effect(se: float, target_power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest true effect a two-sided test of size alpha detects with the target power (one-tail approximation)."""
    from scipy.stats import norm

    return float((norm.ppf(1 - alpha / 2) + norm.ppf(target_power)) * se)


def weighted_groups(values: np.ndarray, weights: np.ndarray, k: int = 5) -> np.ndarray:
    """Group labels 1..k, ordered by `values`, each holding about 1/k of the total weight."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order]) / weights.sum()
    mid = cum - weights[order] / weights.sum() / 2  # weight midpoint of each unit
    out = np.empty(len(values), int)
    out[order] = np.minimum((mid * k).astype(int), k - 1) + 1
    return out
