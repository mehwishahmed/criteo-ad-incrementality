"""Treatment effect estimates: lift, confidence intervals, bootstrap."""
import numpy as np
import pandas as pd
from scipy import stats

Z95 = stats.norm.ppf(0.975)


def two_proportion_lift(df: pd.DataFrame, outcome: str,
                        treatment_col: str = "treatment", alpha: float = 0.05) -> dict:
    """Absolute and relative lift of treatment over control, with normal-approximation CIs.

    - Absolute lift CI: unpooled SE, sqrt(pT(1-pT)/nT + pC(1-pC)/nC).
    - Relative lift CI: delta method on log(pT/pC), so the interval stays asymmetric
      and positive-ratio like the ratio itself.
    - p-value: two-sided z-test with the pooled SE (the SE under H0: pT == pC).
    """
    z = stats.norm.ppf(1 - alpha / 2)
    y_t = df.loc[df[treatment_col] == 1, outcome]
    y_c = df.loc[df[treatment_col] == 0, outcome]
    n_t, n_c = len(y_t), len(y_c)
    p_t, p_c = y_t.mean(), y_c.mean()

    diff = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)

    p_pool = (y_t.sum() + y_c.sum()) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    p_value = 2 * stats.norm.sf(abs(diff / se_pool))

    log_rr = np.log(p_t / p_c)
    se_log_rr = np.sqrt((1 - p_t) / (n_t * p_t) + (1 - p_c) / (n_c * p_c))

    return {
        "metric": outcome,
        "n_treatment": n_t,
        "n_control": n_c,
        "control_rate": p_c,
        "treatment_rate": p_t,
        "abs_lift": diff,
        "abs_ci_low": diff - z * se,
        "abs_ci_high": diff + z * se,
        "rel_lift": p_t / p_c - 1,
        "rel_ci_low": np.exp(log_rr - z * se_log_rr) - 1,
        "rel_ci_high": np.exp(log_rr + z * se_log_rr) - 1,
        "p_value": p_value,
    }


def lift_table(df: pd.DataFrame, outcomes: list[str], **kwargs) -> pd.DataFrame:
    return pd.DataFrame([two_proportion_lift(df, o, **kwargs) for o in outcomes]).set_index("metric")


def rate_ci(y: pd.Series, alpha: float = 0.05) -> tuple[float, float, float]:
    """Rate and normal-approximation CI for a single group's binary outcome."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = y.mean()
    half = z * np.sqrt(p * (1 - p) / len(y))
    return p, p - half, p + half


def stratified_subsample(df: pd.DataFrame, n: int, treatment_col: str = "treatment",
                         seed: int = 42) -> pd.DataFrame:
    """Random subsample of n users that keeps the treatment/control split exactly."""
    frac = n / len(df)
    return (df.groupby(treatment_col, group_keys=False)
              .sample(frac=frac, random_state=seed)
              .reset_index(drop=True))


def bootstrap_lift(df: pd.DataFrame, outcome: str, n_boot: int = 1000,
                   treatment_col: str = "treatment", seed: int = 42) -> dict:
    """Percentile bootstrap CI for absolute and relative lift.

    Resamples users with replacement *within each arm* (keeping arm sizes fixed, as in
    the real experiment), recomputes the lift each time, and takes the 2.5th/97.5th
    percentiles. It makes no normality assumption, so it's a check on the formula CIs.
    """
    rng = np.random.default_rng(seed)
    y_t = df.loc[df[treatment_col] == 1, outcome].to_numpy()
    y_c = df.loc[df[treatment_col] == 0, outcome].to_numpy()

    p_t, p_c = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        p_t[b] = y_t[rng.integers(0, len(y_t), len(y_t))].mean()
        p_c[b] = y_c[rng.integers(0, len(y_c), len(y_c))].mean()
    abs_lift = p_t - p_c
    rel_lift = p_t / p_c - 1

    return {
        "metric": outcome,
        "abs_lift": y_t.mean() - y_c.mean(),
        "abs_ci_low": np.percentile(abs_lift, 2.5),
        "abs_ci_high": np.percentile(abs_lift, 97.5),
        "rel_lift": y_t.mean() / y_c.mean() - 1,
        "rel_ci_low": np.percentile(rel_lift, 2.5),
        "rel_ci_high": np.percentile(rel_lift, 97.5),
        "share_boot_lift_le_0": float((abs_lift <= 0).mean()),
    }


def _p_from_cohens_h(p_base: float, h: float) -> float:
    """Invert Cohen's h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p0)) for p1."""
    return np.sin(np.arcsin(np.sqrt(p_base)) + h / 2) ** 2


def mde_two_proportions(p_control: float, n_treatment: int, n_control: int,
                        alpha: float = 0.05, power: float = 0.80) -> dict:
    """Minimum detectable effect for a two-sided two-proportion test.

    statsmodels solves for the smallest Cohen's h detectable with the given
    sample sizes; we convert h back to a treatment rate to express the MDE in
    percentage points and as a relative lift over the control rate.
    """
    from statsmodels.stats.power import NormalIndPower

    h = NormalIndPower().solve_power(effect_size=None, nobs1=n_treatment,
                                     ratio=n_control / n_treatment,
                                     alpha=alpha, power=power, alternative="two-sided")
    p_t = _p_from_cohens_h(p_control, h)
    return {
        "control_rate": p_control,
        "cohens_h": h,
        "mde_abs": p_t - p_control,
        "mde_rel": p_t / p_control - 1,
    }


def required_total_n(p_control: float, rel_lift: float, treat_share: float = 0.85,
                     alpha: float = 0.05, power: float = 0.80) -> int:
    """Total users needed (at the given split) to detect a relative lift with the given power."""
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    h = proportion_effectsize(p_control * (1 + rel_lift), p_control)
    ratio = (1 - treat_share) / treat_share
    n_t = NormalIndPower().solve_power(effect_size=h, nobs1=None, ratio=ratio,
                                       alpha=alpha, power=power, alternative="two-sided")
    return int(np.ceil(n_t * (1 + ratio)))
