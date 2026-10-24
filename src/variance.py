"""CUPED-style variance reduction via regression adjustment on pre-treatment features."""
import pandas as pd
import statsmodels.api as sm


def treatment_ols(df: pd.DataFrame, outcome: str, covariates: list[str] | None = None,
                  interact: bool = False, treatment_col: str = "treatment") -> dict:
    """OLS of outcome on treatment (+ covariates) with HC1 robust standard errors.

    - No covariates: the treatment coefficient equals the difference in means.
    - Covariates, interact=False: the classic "ANCOVA" adjustment.
    - Covariates, interact=True: Lin (2013) adds treatment x centered-covariate terms.
      This stays consistent for the average treatment effect even when the effect
      varies across users and arms are unequal (85/15), where plain ANCOVA can drift.

    Covariates are mean-centered so the intercept is the (adjusted) control baseline
    for the whole population; centering doesn't change the treatment coefficient.
    HC1 SEs are used because a binary outcome is heteroskedastic by construction.
    """
    t = df[treatment_col].astype("float64")
    parts = [t]
    if covariates:
        cov = df[covariates].astype("float64")
        cov = cov - cov.mean()
        parts.append(cov)
        if interact:
            parts.append(cov.mul(t, axis=0).add_suffix("_x_treatment"))
    X = sm.add_constant(pd.concat(parts, axis=1))
    fit = sm.OLS(df[outcome].astype("float64"), X).fit(cov_type="HC1")

    coef = fit.params[treatment_col]
    lo, hi = fit.conf_int().loc[treatment_col]
    baseline = fit.params["const"]
    return {"coef": coef, "se": fit.bse[treatment_col], "ci_low": lo, "ci_high": hi,
            "ci_width": hi - lo, "control_baseline": baseline, "rel_lift": coef / baseline,
            "r_squared": fit.rsquared}


def compare_adjustment(df: pd.DataFrame, outcome: str, covariates: list[str],
                       treatment_col: str = "treatment") -> pd.DataFrame:
    """Unadjusted vs. ANCOVA vs. Lin-adjusted treatment effect, with % CI-width reduction."""
    fits = {
        "unadjusted": treatment_ols(df, outcome, None, False, treatment_col),
        "adjusted (ANCOVA)": treatment_ols(df, outcome, covariates, False, treatment_col),
        "adjusted (Lin, interacted)": treatment_ols(df, outcome, covariates, True, treatment_col),
    }
    out = pd.DataFrame(fits).T
    naive = fits["unadjusted"]
    out["ci_width_reduction"] = 1 - out["ci_width"] / naive["ci_width"]
    # How much bigger the unadjusted sample would need to be to match this precision
    out["equivalent_sample_multiplier"] = (naive["se"] / out["se"]) ** 2
    return out
