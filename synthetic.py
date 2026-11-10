"""Synthetic data for the CASCADE-CRC experiments.

This module only generates data. Every model fit, calibration step and method lives in
cascade_crc.ipynb.

A *world* fixes the coefficient vectors. Each replication draws four independent samples
from it, one per role in the protocol:

    A  fit set          (2N rows)  point predictor and Stage 1 classifier
    B  development set  (2N rows)  Venn-Abers calibration, difficulty and scale models
    C  calibration set  ( N rows)  conformal calibration
    T  evaluation set   (4000 rows)

Outcomes are zero-inflated: Z ~ Bernoulli(logistic(X @ wz)); y = 0 when Z = 0, otherwise
mu + e (signed) or |mu + e| + 0.1 (nonnegative). The noise scale is sigma = exp(het * d),
where d mixes the Stage 1 direction wz with an independent direction wd:

    d = relev * (X @ wz) + sqrt(1 - relev^2) * (X @ wd)

so `relev` controls whether Stage 1 information is informative (> 0), irrelevant (0) or
misleading (< 0) about difficulty. Noise is Student-t with 5 degrees of freedom, rescaled to
unit variance.

Seeds: the world uses `rep`; the four samples use 500000+rep, 600000+rep, 700000+rep and
800000+rep. These are the seeds the paper's runs used.
"""
import numpy as np

N_FEATURES = 6
N_EVAL = 4000


def world(seed, het, relev, signed):
    """Coefficients for one replication."""
    r = np.random.default_rng(seed)
    return dict(wz=r.normal(size=N_FEATURES) / np.sqrt(N_FEATURES),
                wm=r.normal(size=N_FEATURES) / np.sqrt(N_FEATURES),
                wd=r.normal(size=N_FEATURES) / np.sqrt(N_FEATURES),
                het=het, relev=relev, signed=signed)


def draw(W, n, seed):
    """n rows from world W. Returns X, y, Z and the true noise scale sigma."""
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, N_FEATURES))
    Z = (r.random(n) < 1 / (1 + np.exp(-(X @ W["wz"])))).astype(int)
    d = W["relev"] * (X @ W["wz"]) + np.sqrt(max(1 - W["relev"] ** 2, 0)) * (X @ W["wd"])
    sigma = np.exp(W["het"] * d)
    e = sigma * r.standard_t(5, n) / np.sqrt(5 / 3)
    mu = X @ W["wm"]
    y = np.where(Z == 1, mu + e if W["signed"] else np.abs(mu + e) + 0.1, 0.0)
    return X, y, Z, sigma


def replication(N, het, relev, signed, rep):
    """All four samples of one replication, stacked into single arrays.

    Returns a dict with X, y, z, sigma (one row per drawn unit, in the order A, B, C, T)
    and `roles`, a dict mapping "A", "B", "C", "T" to the row indices of each sample.
    """
    W = world(rep, het, relev, signed)
    parts = [("A", 2 * N, 500000 + rep), ("B", 2 * N, 600000 + rep),
             ("C", N, 700000 + rep), ("T", N_EVAL, 800000 + rep)]
    X, y, z, sig, roles, start = [], [], [], [], {}, 0
    for name, n, seed in parts:
        Xi, yi, zi, si = draw(W, n, seed)
        X.append(Xi); y.append(yi); z.append(zi); sig.append(si)
        roles[name] = np.arange(start, start + n)
        start += n
    return dict(X=np.vstack(X), y=np.concatenate(y), z=np.concatenate(z),
                sigma=np.concatenate(sig), roles=roles)
