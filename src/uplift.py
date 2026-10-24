"""Heterogeneous effects: T-learner uplift model and Qini-based evaluation."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.variance import treatment_ols

XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, random_state=42, n_jobs=-1)


def split(df: pd.DataFrame, test_size: float = 0.3, treatment_col: str = "treatment",
          seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """70/30 train/test split, stratified by treatment so both sets keep the 85/15 ratio."""
    return train_test_split(df, test_size=test_size, stratify=df[treatment_col], random_state=seed)


class TLearner:
    """Two separate outcome models: one on treated users, one on control users.

    uplift(x) = P(outcome | x, treated) - P(outcome | x, control)
    """

    def __init__(self, **params):
        self.params = {**XGB_PARAMS, **params}

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "TLearner":
        self.model_t = XGBClassifier(**self.params).fit(X[t == 1], y[t == 1])
        self.model_c = XGBClassifier(**self.params).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_t.predict_proba(X)[:, 1] - self.model_c.predict_proba(X)[:, 1]


def targeting_table(y: pd.Series, uplift: np.ndarray, t: pd.Series,
                    fractions=(0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Share of incremental outcomes captured by targeting the top X% by predicted uplift.

    Incremental outcomes in the targeted group use the Qini definition:
    treated outcomes - control outcomes rescaled to the treated count,
        Q(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k),
    and the share is Q(k) / Q(all users). Random targeting captures X% on average.
    """
    order = np.argsort(-uplift, kind="stable")
    y, t = y.to_numpy()[order], t.to_numpy()[order]

    def qini(n):
        yt, yc = y[:n][t[:n] == 1], y[:n][t[:n] == 0]
        return yt.sum() - yc.sum() * len(yt) / len(yc)

    total = qini(len(y))
    rows = []
    for f in fractions:
        n = int(round(f * len(y)))
        rows.append({"target top": f, "users targeted": n,
                     "incremental visits (Qini units)": qini(n),
                     "share of incremental captured": qini(n) / total,
                     "random targeting share": f,
                     "lift over random": qini(n) / total / f})
    return pd.DataFrame(rows).set_index("target top")


def adjusted_targeting_table(df: pd.DataFrame, uplift: np.ndarray, outcome: str,
                             covariates: list[str], fractions=(0.1, 0.2, 0.3),
                             treatment_col: str = "treatment") -> pd.DataFrame:
    """Same question as targeting_table, but each group's incremental outcomes are
    Lin-adjusted ITT x group size, so the pre-treatment imbalance from Step 5 doesn't
    inflate the counts. Share = adjusted incremental in top X% / adjusted incremental overall.
    """
    order = np.argsort(-uplift, kind="stable")
    ranked = df.iloc[order]
    total = treatment_ols(ranked, outcome, covariates, interact=True, treatment_col=treatment_col)["coef"] * len(ranked)
    rows = []
    for f in fractions:
        top = ranked.iloc[: int(round(f * len(ranked)))]
        fit = treatment_ols(top, outcome, covariates, interact=True, treatment_col=treatment_col)
        rows.append({"target top": f, "adjusted ITT in group (pp)": fit["coef"] * 100,
                     "adjusted share of incremental captured": fit["coef"] * len(top) / total,
                     "random targeting share": f})
    return pd.DataFrame(rows).set_index("target top")


def seed_stability(df: pd.DataFrame, features: list[str], outcome: str = "visit",
                   seeds=(42, 43, 44, 45, 46), fractions=(0.1, 0.2, 0.3),
                   treatment_col: str = "treatment") -> pd.DataFrame:
    """Refit the T-learner with a different train/test split *and* XGBoost seed each time.

    Returns one row per seed with the Qini AUC and targeting shares (standard Qini and
    Lin-adjusted), so we can quote a mean and range instead of one split's numbers.
    """
    from sklift.metrics import qini_auc_score

    rows = []
    for s in seeds:
        train, test = split(df, seed=s, treatment_col=treatment_col)
        model = TLearner(random_state=s).fit(train[features], train[outcome], train[treatment_col])
        up = model.predict(test[features])
        qini = targeting_table(test[outcome], up, test[treatment_col], fractions)
        adj = adjusted_targeting_table(test, up, outcome, features, fractions, treatment_col)
        row = {"seed": s, "qini_auc": qini_auc_score(test[outcome], up, test[treatment_col])}
        for f in fractions:
            row[f"top{int(f * 100)}_qini"] = qini.loc[f, "share of incremental captured"]
            row[f"top{int(f * 100)}_adjusted"] = adj.loc[f, "adjusted share of incremental captured"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("seed")
