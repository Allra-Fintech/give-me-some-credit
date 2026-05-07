from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def compare_models(df: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
    X = df[FEATURE_COLS]
    y = df[TARGET]
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    candidates = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="auc", verbosity=0),
    }

    rows = []
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X, y, cv=cv_split, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "auc_mean": scores.mean(), "auc_std": scores.std()})

    return (
        pd.DataFrame(rows)
        .sort_values("auc_mean", ascending=False)
        .reset_index(drop=True)
    )


def calibration_curves(df: pd.DataFrame, n_bins: int = 10) -> dict[str, tuple]:
    X = df[FEATURE_COLS]
    y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="auc", verbosity=0),
    }

    curves = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_val, proba, n_bins=n_bins)
        curves[name] = (frac_pos, mean_pred)

    return curves
