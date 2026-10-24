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


def prognostic_balance(df: pd.DataFrame, features: list[str], outcome: str,
                       treatment_col: str = "treatment") -> dict:
    """Balance on a single 'prognostic score': the outcome predicted from features.

    Fit outcome ~ features on control users only (so treatment can't leak in), predict
    for everyone, and compare the mean prediction by arm. This folds all feature
    imbalances into one number in outcome units: "how much higher would treatment's
    outcome be from its features alone, with no ads?" Also reports the z-score of the
    gap vs. what randomization alone would produce.
    """
    import statsmodels.api as sm

    X = sm.add_constant(df[features].astype("float64"))
    is_c = df[treatment_col] == 0
    fit = sm.OLS(df.loc[is_c, outcome].astype("float64"), X[is_c]).fit()
    score = fit.predict(X)
    m_t, m_c = score[~is_c].mean(), score[is_c].mean()
    se = np.sqrt(score[~is_c].var() / (~is_c).sum() + score[is_c].var() / is_c.sum())
    return {"pred_treatment": m_t, "pred_control": m_c, "gap": m_t - m_c,
            "smd": (m_t - m_c) / score.std(), "z": (m_t - m_c) / se}


def smd_z_scores(smd: pd.DataFrame, n_treatment: int, n_control: int) -> pd.Series:
    """SMD divided by its standard error under pure randomization, sqrt(1/nT + 1/nC)."""
    return smd.set_index("feature")["smd"] / np.sqrt(1 / n_treatment + 1 / n_control)
