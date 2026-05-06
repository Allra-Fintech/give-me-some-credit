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
# # 01 — Exploratory Data Analysis
#
# **Goal:** Understand the raw data shape, missing values, and label distribution.

# %% tags=["parameters"]
data_dir: str = "../data"
output_path: str = "../outputs/01_eda.ipynb"

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from give_me_some_credit.data import load_train

sns.set_theme(style="whitegrid")

# %%
df = load_train()
print(df.shape)
df.head()

# %% [markdown]
# ## Missing values

# %%
missing = df.isnull().mean().sort_values(ascending=False)
missing[missing > 0]

# %% [markdown]
# ## Target distribution

# %%
df["serious_dlqin2yrs"].value_counts(normalize=True).rename("proportion").to_frame()

# %% [markdown]
# ## Numeric distributions

# %%
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, df.select_dtypes("number").columns):
    df[col].dropna().hist(ax=ax, bins=40)
    ax.set_title(col, fontsize=8)
fig.tight_layout()
plt.show()
