"""Non-compliance: intent-to-treat (ITT) vs. complier average causal effect (CACE)."""
import numpy as np
import pandas as pd

from src.variance import treatment_ols


def wald_cace(df: pd.DataFrame, outcome: str, treatment_col: str = "treatment",
              exposure_col: str = "exposure") -> dict:
    """Wald / IV estimator: CACE = ITT on outcome / ITT on exposure (difference in means)."""
    t = df[treatment_col] == 1
    itt_y = df.loc[t, outcome].mean() - df.loc[~t, outcome].mean()
    itt_d = df.loc[t, exposure_col].mean() - df.loc[~t, exposure_col].mean()
    return {"itt_outcome": itt_y, "itt_exposure": itt_d, "cace": itt_y / itt_d}


def bootstrap_cace(df: pd.DataFrame, outcome: str, n_boot: int = 1000, seed: int = 42,
                   treatment_col: str = "treatment", exposure_col: str = "exposure") -> dict:
    """Percentile bootstrap CI for the Wald CACE, resampling users within each arm.

    Resampling a user keeps their outcome and exposure together, so the bootstrap
    captures the correlation between numerator and denominator of the ratio.
    """
    rng = np.random.default_rng(seed)
    t = (df[treatment_col] == 1).to_numpy()
    y, d = df[outcome].to_numpy(), df[exposure_col].to_numpy()
    y_t, d_t, y_c, d_c = y[t], d[t], y[~t], d[~t]

    draws = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y_t), len(y_t))
        j = rng.integers(0, len(y_c), len(y_c))
        draws[b] = (y_t[i].mean() - y_c[j].mean()) / (d_t[i].mean() - d_c[j].mean())
    return {"ci_low": np.percentile(draws, 2.5), "ci_high": np.percentile(draws, 97.5),
            "boot_se": draws.std(ddof=1)}


def adjusted_cace(df: pd.DataFrame, outcome: str, covariates: list[str],
                  treatment_col: str = "treatment", exposure_col: str = "exposure") -> dict:
    """Covariate-adjusted CACE: Lin-adjusted ITT on outcome / Lin-adjusted ITT on exposure.

    SE via the standard IV linearization: at the estimate c, the ITT of the combined
    outcome (y - c * exposure) is zero, and SE(c) = SE(ITT of y - c*d) / ITT_d. Fitting
    that one regression gives the joint uncertainty of numerator and denominator.
    """
    itt_y = treatment_ols(df, outcome, covariates, interact=True, treatment_col=treatment_col)
    itt_d = treatment_ols(df, exposure_col, covariates, interact=True, treatment_col=treatment_col)
    c = itt_y["coef"] / itt_d["coef"]

    combined = df.assign(_resid=df[outcome] - c * df[exposure_col])
    se = treatment_ols(combined, "_resid", covariates, interact=True,
                       treatment_col=treatment_col)["se"] / itt_d["coef"]
    return {"itt_outcome": itt_y["coef"], "itt_exposure": itt_d["coef"], "cace": c,
            "se": se, "ci_low": c - 1.96 * se, "ci_high": c + 1.96 * se}


def complier_means(df: pd.DataFrame, outcome: str, treatment_col: str = "treatment",
                   exposure_col: str = "exposure") -> dict:
    """Outcome rates for compliers with and without ads, under one-sided non-compliance.

    Treated & exposed users are compliers, so E[Y1 | complier] is observed directly.
    Control users are a random mix of compliers (share pi) and never-takers, and
    never-takers' no-ad outcome is seen in treated-but-unexposed users, so
    E[Y0 | complier] = (E[Y | control] - (1 - pi) * E[Y | treated, unexposed]) / pi.
    """
    t = df[treatment_col] == 1
    exposed = df[exposure_col] == 1
    pi = exposed[t].mean()
    y1_c = df.loc[t & exposed, outcome].mean()
    y_never = df.loc[t & ~exposed, outcome].mean()
    y0_c = (df.loc[~t, outcome].mean() - (1 - pi) * y_never) / pi
    return {"complier_share": pi, "compliers_with_ads": y1_c, "compliers_without_ads": y0_c,
            "never_takers": y_never, "naive_exposed_vs_unexposed": y1_c - y_never}
