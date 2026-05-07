from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit.features.engineering import TARGET

FEATURE_COLS = [
    "revolving_utilization_of_unsecured_lines",
    "age",
    "number_of_time30_59_days_past_due_not_worse",
    "debt_ratio_clipped",
    "monthly_income",
    "number_of_open_credit_lines_and_loans",
    "number_of_times90_days_late",
    "number_real_estate_loans_or_lines",
    "number_of_time60_89_days_past_due_not_worse",
    "number_of_dependents",
    "income_per_dependent",
    "total_past_due",
    "log_revolving_utilization",
]


def train(df: pd.DataFrame, **kwargs) -> GradientBoostingClassifier:
    X = df[FEATURE_COLS]
    y = df[TARGET]
    params = {"n_estimators": 100, "max_depth": 4, "random_state": 42, **kwargs}
    clf = GradientBoostingClassifier(**params)
    clf.fit(X, y)
    return clf


def evaluate(clf: GradientBoostingClassifier, df: pd.DataFrame) -> dict[str, float]:
    X = df[FEATURE_COLS]
    y = df[TARGET]
    proba = clf.predict_proba(X)[:, 1]
    return {"roc_auc": roc_auc_score(y, proba)}
