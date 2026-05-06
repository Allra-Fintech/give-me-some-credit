# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Model Training
#
# Trains a GradientBoostingClassifier on the full training set and reports ROC-AUC.

# %% tags=["parameters"]
n_estimators: int = 100
max_depth: int = 4
output_path: str = "../outputs/03_model_training.ipynb"

# %%
import joblib

from credit.data import load_train
from credit.features import build_features
from credit.models import evaluate, train
from credit.utils import get_logger

log = get_logger("03_model_training")

# %%
df = build_features(load_train())
clf = train(df, n_estimators=n_estimators, max_depth=max_depth)

metrics = evaluate(clf, df)
log.info("Train metrics: %s", metrics)
metrics

# %%
joblib.dump(clf, "../outputs/model.joblib")
log.info("Model saved.")
