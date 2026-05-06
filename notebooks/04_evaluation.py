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
# # 04 — Evaluation
#
# Loads a trained model, runs it on the test set, and produces a submission file.

# %% tags=["parameters"]
model_path: str = "../outputs/model.joblib"
output_path: str = "../outputs/04_evaluation.ipynb"
submission_path: str = "../outputs/submission.csv"

# %%
import joblib
import pandas as pd
from sklearn.metrics import RocCurveDisplay

from credit.data import load_test, load_train
from credit.features import build_features
from credit.models.train import FEATURE_COLS

clf = joblib.load(model_path)

# %%
test = build_features(load_test())
proba = clf.predict_proba(test[FEATURE_COLS])[:, 1]

submission = pd.DataFrame({"Id": test.index, "Probability": proba})
submission.to_csv(submission_path, index=False)
print(f"Submission saved: {submission.shape}")
submission.head()
