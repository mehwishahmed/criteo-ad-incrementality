"""CUPED-style variance reduction via regression adjustment on pre-treatment features."""
import numpy as np
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


def cross_fit_propensity(df: pd.DataFrame, covariates: list[str], treatment_col: str = "treatment",
                         n_folds: int = 5, seed: int = 42) -> pd.Series:
    """Out-of-fold P(treatment | features) from XGBoost.

    Each user's score comes from a model that never saw them, so the propensity AUC
    is an honest measure of how predictable assignment is from features (0.5 = not at all).
    """
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    e = np.empty(len(df))
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    X, t = df[covariates], df[treatment_col]
    for train, test in folds.split(X, t):
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              random_state=seed, n_jobs=-1)
        model.fit(X.iloc[train], t.iloc[train])
        e[test] = model.predict_proba(X.iloc[test])[:, 1]
    return pd.Series(e, index=df.index, name="propensity")


def ipw_effect(df: pd.DataFrame, outcome: str, propensity: pd.Series,
               treatment_col: str = "treatment") -> dict:
    """Inverse-propensity-weighted (Hajek) average treatment effect.

    Each treated user gets weight 1/e(x) and each control user 1/(1 - e(x)), which
    reweights both arms to look like the full population on the features. The SE comes
    from a weighted regression with HC1 errors, treating the propensities as fixed
    (a standard approximation; it slightly overstates uncertainty).
    """
    t = df[treatment_col].astype("float64")
    w = t / propensity + (1 - t) / (1 - propensity)
    fit = sm.WLS(df[outcome].astype("float64"), sm.add_constant(t), weights=w).fit(cov_type="HC1")
    coef, base = fit.params[treatment_col], fit.params["const"]
    lo, hi = fit.conf_int().loc[treatment_col]
    return {"coef": coef, "se": fit.bse[treatment_col], "ci_low": lo, "ci_high": hi,
            "control_baseline": base, "rel_lift": coef / base,
            "max_weight": float(w.max()), "effective_n": float(w.sum() ** 2 / (w ** 2).sum())}


def aipw_effect(df: pd.DataFrame, outcome: str, covariates: list[str], propensity: pd.Series,
                treatment_col: str = "treatment", n_folds: int = 5, clip: float = 0.01,
                seed: int = 42) -> dict:
    """Doubly robust (augmented IPW) average treatment effect with cross-fitting.

    psi_i = mu1(x) - mu0(x) + T (y - mu1(x)) / e(x) - (1 - T) (y - mu0(x)) / (1 - e(x))
    The estimate is mean(psi), SE = sd(psi) / sqrt(n). It's consistent if *either* the
    outcome models mu_t (XGBoost, fit out-of-fold) or the propensity e(x) is right, and
    the outcome-model term absorbs most of the noise that makes plain IPW unstable.
    Propensities are clipped to [clip, 1 - clip] to bound the weights.
    """
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    X, t, y = df[covariates], df[treatment_col].to_numpy(), df[outcome].to_numpy()
    mu = np.empty((len(df), 2))
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for train, test in folds.split(X, t):
        for arm in (0, 1):
            idx = train[t[train] == arm]
            model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=seed, n_jobs=-1)
            model.fit(X.iloc[idx], y[idx])
            mu[test, arm] = model.predict_proba(X.iloc[test])[:, 1]

    e = propensity.clip(clip, 1 - clip).to_numpy()
    psi = (mu[:, 1] - mu[:, 0] + t * (y - mu[:, 1]) / e - (1 - t) * (y - mu[:, 0]) / (1 - e))
    psi0 = mu[:, 0] + (1 - t) * (y - mu[:, 0]) / (1 - e)
    coef, se = psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))
    return {"coef": coef, "se": se, "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se,
            "control_baseline": psi0.mean(), "rel_lift": coef / psi0.mean()}
