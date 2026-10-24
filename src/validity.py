"""Experiment validity checks: SRM, covariate balance, exposure."""
import numpy as np
import pandas as pd
from scipy import stats


def srm_test(df: pd.DataFrame, expected_treat_share: float = 0.85,
             treatment_col: str = "treatment") -> dict:
    """Chi-square goodness-of-fit test of treatment/control counts vs. the intended split.

    A tiny p-value (convention: < 0.001) signals sample ratio mismatch, i.e. the
    assignment mechanism or data pipeline is broken and results can't be trusted.
    """
    n = len(df)
    n_treat = int(df[treatment_col].sum())
    observed = np.array([n_treat, n - n_treat])
    expected = np.array([expected_treat_share, 1 - expected_treat_share]) * n
    chi2, p = stats.chisquare(observed, expected)
    return {
        "n": n,
        "n_treatment": n_treat,
        "n_control": n - n_treat,
        "observed_treat_share": n_treat / n,
        "expected_treat_share": expected_treat_share,
        "chi2": chi2,
        "p_value": p,
    }


def standardized_mean_diff(df: pd.DataFrame, features: list[str],
                           treatment_col: str = "treatment") -> pd.DataFrame:
    """SMD = (mean_T - mean_C) / sqrt((var_T + var_C) / 2) for each feature.

    Unlike a t-test, SMD doesn't shrink toward "significant" as n grows, so it
    measures whether the imbalance is large enough to matter (|SMD| > 0.1 rule).
    """
    t = df[df[treatment_col] == 1]
    c = df[df[treatment_col] == 0]
    rows = []
    for f in features:
        mt, mc = t[f].mean(), c[f].mean()
        pooled_sd = np.sqrt((t[f].var() + c[f].var()) / 2)
        rows.append({"feature": f, "mean_treatment": mt, "mean_control": mc,
                     "smd": (mt - mc) / pooled_sd})
    out = pd.DataFrame(rows)
    out["imbalanced"] = out["smd"].abs() > 0.1
    return out


def exposure_check(df: pd.DataFrame, treatment_col: str = "treatment",
                   exposure_col: str = "exposure") -> pd.DataFrame:
    """Exposure rate by arm. One-sided non-compliance means control exposure == 0."""
    return (df.groupby(treatment_col)[exposure_col]
              .agg(n="size", n_exposed="sum", exposure_rate="mean")
              .rename(index={0: "control", 1: "treatment"}))
