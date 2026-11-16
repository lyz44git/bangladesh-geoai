"""Quantitative spatial equilibrium model (Redding 2016 type) with exact hat algebra.

Economy
  * N locations. Each produces CES varieties (elasticity sigma). The number of
    varieties is proportional to population (Krugman), which is the
    agglomeration force.
  * Housing is in fixed supply and takes a share (1 - alpha) of spending. This
    is the dispersion force.
  * Optional local non-traded sector (alpha_N > 0): a share alpha_N of spending
    goes to goods and services that are produced and consumed within the
    location (retail, transport, education, health, personal services) with
    labour only, so their price moves one-for-one with the local wage. The
    traded share is alpha_T = alpha - alpha_N. With free entry, revenue equals
    the wage bill in both sectors, so the non-traded sector employs a fixed
    share alpha_N / alpha of local workers, traded varieties remain proportional
    to population, and the goods-market clearing condition is unchanged.
  * Workers choose where to live. Idiosyncratic tastes are Frechet with shape
    eps (the migration elasticity). eps = None means no mobility (short run).
  * Iceberg trade costs: d_ij^(1-sigma) = t_ij^(-kappa), with t_ij the travel
    time in hours, so kappa is the travel-time elasticity of trade.

Equilibrium in changes (x_hat = x'/x), given the baseline trade shares pi_ij
(origin i, destination j; columns sum to 1), income Y_i and population shares
lam_i:
  P_hat_j^(1-sigma) = sum_k pi_kj L_hat_k w_hat_k^(1-sigma) phi_hat_kj
  pi'_ij            = pi_ij L_hat_i w_hat_i^(1-sigma) phi_hat_ij / P_hat_j^(1-sigma)
  w_hat_i L_hat_i Y_i = sum_j pi'_ij w_hat_j L_hat_j Y_j        (goods market clearing)
  V_hat_i           = w_hat_i^alpha_T P_hat_i^(-alpha_T) L_hat_i^(-(1-alpha))
                      (non-traded prices move with wages and cancel against
                      them; alpha_T = alpha when alpha_N = 0)
  L_hat_i           = V_hat_i^eps / sum_k lam_k V_hat_k^eps     (mobility)
  welfare           = (sum_k lam_k V_hat_k^eps)^(1/eps)
With no mobility, welfare is the population-weighted mean of V_hat.

Hat algebra means unobserved productivities and amenities never need to be
recovered separately: they are absorbed in the baseline shares.
"""
from __future__ import annotations

import numpy as np


def trade_shares(Y: np.ndarray, phi: np.ndarray, tol: float = 1e-10, max_iter: int = 100000):
    """Baseline trade shares consistent with observed incomes.

    pi_ij = S_i phi_ij / sum_k S_k phi_kj must clear every goods market,
    sum_j pi_ij Y_j = Y_i. Writing X_ij = a_i phi_ij b_j (with a = S and
    b_j = Y_j / Phi_j) this is a matrix-scaling problem: X must have row sums Y
    (sales) and column sums Y (spending). Solved with Sinkhorn-Knopp. Returns
    (pi, S) with S normalised to sum to one.
    """
    a, b = np.ones_like(Y), np.ones_like(Y)
    for _ in range(max_iter):
        a = Y / (phi @ b)
        b = Y / (phi.T @ a)
        if np.max(np.abs(a * (phi @ b) / Y - 1)) < tol:
            break
    X = a[:, None] * phi * b[None, :]
    return X / X.sum(0, keepdims=True), a / a.sum()


def counterfactual(pi, Y, lam, phi_hat, sigma: float, alpha: float, eps: float | None = None,
                   alpha_N: float = 0.0, damp: float = 0.5, tol: float = 1e-8, max_iter: int = 20000) -> dict:
    """Solve for the change in wages, population, prices and welfare after a trade-cost shock.

    phi_hat = d_hat^(1-sigma). Aggregate nominal income is the numeraire.
    alpha is the non-housing spending share; alpha_N the part of it spent on
    local non-traded goods (0 = one-sector model). P_hat is the traded-goods
    price index.
    """
    a_T = alpha - alpha_N
    n = len(Y)
    w, L = np.ones(n), np.ones(n)
    for it in range(max_iter):
        x = pi * (L * w ** (1 - sigma))[:, None] * phi_hat
        Pp = x.sum(0)
        income = w * L * Y
        z = ((x / Pp) @ income) / income  # sales / income: excess demand for i's goods
        # Tatonnement: log sales respond to log wages with elasticity about -sigma,
        # so a step of 1/sigma in logs is close to a Newton step.
        w = w * z ** (1 / sigma)
        w *= Y.sum() / (w * L * Y).sum()
        err = np.abs(np.log(z)).max()
        if eps is not None:
            V = w ** a_T * Pp ** (-a_T / (1 - sigma)) * L ** (-(1 - alpha))
            L_new = V ** eps / (lam * V ** eps).sum()
            err = max(err, np.abs(np.log(L_new / L)).max())
            L = L * (L_new / L) ** damp
        if err < tol:
            break
    x = pi * (L * w ** (1 - sigma))[:, None] * phi_hat
    P = x.sum(0) ** (1 / (1 - sigma))
    V = w ** a_T * P ** (-a_T) * L ** (-(1 - alpha))
    welfare = float((lam * V).sum()) if eps is None else float((lam * V ** eps).sum() ** (1 / eps))
    return dict(w_hat=w, L_hat=L, P_hat=P, V_hat=V, welfare=welfare, iterations=it + 1, converged=err < tol)


def market_access(phi: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Reduced-form market access sum_j phi_ij Y_j (Donaldson-Hornbeck style, first order)."""
    return phi @ Y
