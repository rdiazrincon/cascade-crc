"""Evaluation metrics for the CASCADE-CRC experiments.

Interval methods live in cascade_crc.ipynb; this module only scores intervals.

Definitions used throughout the paper, for a target coverage 1 - alpha and G groups:

    per-group coverage  c_j  = mean of the coverage indicator among evaluation units in group j
    CovGap                   = mean over non-empty groups of |c_j - (1 - alpha)|
    FSC (worst group)        = min over non-empty groups of c_j

Clinical results are stored one row per evaluated unit, so the paper's numbers are computed
within each split and then averaged over splits (`split_summary`).
"""
import numpy as np
import pandas as pd
from scipy.stats import beta as _beta


# ---------------------------------------------------------------- per-split quantities
def group_coverage(covered, groups, G):
    """Coverage within each of G groups; NaN for a group with no units."""
    covered = np.asarray(covered, dtype=float)
    return np.array([covered[groups == j].mean() if (groups == j).any() else np.nan
                     for j in range(G)])


def group_mean(values, groups, G):
    """Mean of `values` within each of G groups; NaN for a group with no units."""
    values = np.asarray(values, dtype=float)
    return np.array([values[groups == j].mean() if (groups == j).any() else np.nan
                     for j in range(G)])


def covgap(per_group, alpha):
    """Mean absolute deviation of group coverage from the target, over non-empty groups."""
    return float(np.nanmean(np.abs(np.asarray(per_group) - (1 - alpha))))


def fsc(per_group):
    """Worst-group coverage over non-empty groups."""
    return float(np.nanmin(per_group))


def interval_length(lo, hi):
    """Length of each interval, with empty intervals (hi < lo) counted as zero."""
    return np.maximum(np.asarray(hi) - np.asarray(lo), 0.0)


# ---------------------------------------------------------------- unit-level results
def split_summary(units, alpha, group_col="g_d2", G=3):
    """Split-averaged coverage, length, CovGap and FSC for every arm.

    `units` holds one row per evaluated unit per split per arm, with columns `split`, `arm`,
    `covered`, `set_length` and the group column. Returns one row per arm.
    """
    rows = []
    for arm, a in units.groupby("arm", sort=False):
        per = []
        for _, s in a.groupby("split"):
            g = group_coverage(s["covered"].values, s[group_col].values, G)
            per.append(dict(coverage=s["covered"].mean(), length=s["set_length"].mean(),
                            covgap=covgap(g, alpha), fsc=fsc(g)))
        per = pd.DataFrame(per)
        rows.append(dict(arm=arm, n_splits=len(per), **per.mean().to_dict()))
    return pd.DataFrame(rows)


def per_split_covgap(units, alpha, group_col="g_d2", G=3):
    """CovGap within each split for one arm, indexed by split."""
    return units.groupby("split").apply(
        lambda s: covgap(group_coverage(s["covered"].values, s[group_col].values, G), alpha))


def group_sizes(units, group_col="g_d2", G=3):
    """Evaluation units per group in each split (the n behind each c_j)."""
    one_arm = units[units.arm == units.arm.iloc[0]]
    return (one_arm.groupby(["split", group_col]).size().unstack(fill_value=0)
            .reindex(columns=range(G), fill_value=0))


# ---------------------------------------------------------------- uncertainty
def clopper_pearson(k, n, level=0.95):
    """Two-sided Clopper-Pearson interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    a = 1 - level
    lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi
