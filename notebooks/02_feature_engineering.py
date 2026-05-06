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
# # 02 — Feature Engineering
#
# Applies `build_features` from the package and inspects the result.

# %% tags=["parameters"]
output_path: str = "../outputs/02_feature_engineering.ipynb"

# %%
import pandas as pd

from credit.data import load_train
from credit.features import build_features

# %%
raw = load_train()
featured = build_features(raw)

print(f"Columns added: {set(featured.columns) - set(raw.columns)}")
featured.describe().T

# %% [markdown]
# ## Correlation with target

# %%
import matplotlib.pyplot as plt
import seaborn as sns

corr = featured.corr(numeric_only=True)["serious_dlqin2yrs"].drop("serious_dlqin2yrs").sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
corr.plot(kind="barh", ax=ax)
ax.set_title("Feature correlation with target")
plt.tight_layout()
plt.show()
