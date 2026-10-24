"""Segment-level treatment effects with multiple-testing correction."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.variance import treatment_ols


def quantile_buckets(s: pd.Series, q: int = 5) -> pd.Series:
    """Quantile buckets labeled by value range. Ties at a mass point stay in one bucket,
    so buckets can be uneven (and fewer than q) for features with repeated values."""
    b = pd.qcut(s, q, duplicates="drop")
    return b.map(lambda iv: f"{s.name} ({iv.left:.2f}, {iv.right:.2f}]").astype(str)


def segment_effects(df: pd.DataFrame, segment_features: list[str], covariates: list[str],
                    outcome: str = "visit", q: int = 5,
                    treatment_col: str = "treatment") -> pd.DataFrame:
    """Lin-adjusted treatment effect within each quantile bucket of each segment feature.

    Two hypotheses per segment:
    - p_value: is the effect in this segment different from zero?
    - p_diff_overall: is this segment's effect different from the overall effect?
      (z-test on the difference; treats the overall estimate as fixed, a slight
      approximation since the segment is part of it.)
    """
    overall = treatment_ols(df, outcome, covariates, interact=True, treatment_col=treatment_col)
    rows = []
    for feat in segment_features:
        for label, g in df.groupby(quantile_buckets(df[feat], q), sort=False):
            # A feature can be constant inside a bucket (mass points); drop it there
            covs = [c for c in covariates if g[c].nunique() > 1]
            fit = treatment_ols(g, outcome, covs, interact=True, treatment_col=treatment_col)
            raw = g.loc[g[treatment_col] == 1, outcome].mean() - g.loc[g[treatment_col] == 0, outcome].mean()
            rows.append({
                "feature": feat, "segment": label, "lower": g[feat].min(), "n": len(g),
                "n_control": int((g[treatment_col] == 0).sum()),
                "raw_lift": raw, "adj_lift": fit["coef"], "se": fit["se"],
                "ci_low": fit["ci_low"], "ci_high": fit["ci_high"], "rel_lift": fit["rel_lift"],
                "p_value": 2 * stats.norm.sf(abs(fit["coef"] / fit["se"])),
                "p_diff_overall": 2 * stats.norm.sf(abs((fit["coef"] - overall["coef"]) / fit["se"])),
            })
    out = pd.DataFrame(rows).sort_values(["feature", "lower"]).reset_index(drop=True)
    return out.drop(columns="lower")


def correct(p: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Holm (controls family-wise error) and Benjamini-Hochberg (controls false discovery rate)."""
    holm_rej, holm_p, _, _ = multipletests(p, alpha=alpha, method="holm")
    bh_rej, bh_p, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return pd.DataFrame({"p_raw": p.to_numpy(), "p_holm": holm_p, "p_bh": bh_p,
                         "sig_raw": p.to_numpy() < alpha, "sig_holm": holm_rej, "sig_bh": bh_rej},
                        index=p.index)
