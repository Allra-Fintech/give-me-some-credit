import numpy as np
import pandas as pd


TARGET = "serious_dlqin2yrs"

FILL_MEDIAN = [
    "monthly_income",
    "number_of_dependents",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in FILL_MEDIAN:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Derived ratios
    df["debt_ratio_clipped"] = df["debt_ratio"].clip(0, 10)
    df["income_per_dependent"] = df["monthly_income"] / (df["number_of_dependents"] + 1)
    df["total_past_due"] = (
        df["number_of_times90_days_late"]
        + df["number_of_time60_89_days_past_due_not_worse"]
        + df["number_of_time30_59_days_past_due_not_worse"]
    )
    df["log_revolving_utilization"] = np.log1p(
        df["revolving_utilization_of_unsecured_lines"].clip(0, None)
    )

    return df
