# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

from credit.data import load_train

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
# ## Summary statistics

# %%
df.describe().T

# %% [markdown]
# ## Duplicate rows

# %%
print(f"Duplicate rows: {df.duplicated().sum()}")

# %% [markdown]
# ## Numeric distributions

# %%
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, df.select_dtypes("number").columns):
    data = df[col].dropna()
    data.hist(ax=ax, bins=40)
    ax.set_xlim(data.min(), data.max())
    ax.set_title(col, fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Outliers (IQR method)

# %%
num_cols = df.select_dtypes("number").columns.drop("serious_dlqin2yrs")
q1, q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
iqr = q3 - q1
outlier_pct = ((df[num_cols] < q1 - 1.5 * iqr) | (df[num_cols] > q3 + 1.5 * iqr)).mean()
outlier_pct.sort_values(ascending=False).rename("outlier_rate").to_frame()

# %% [markdown]
# ## Correlation heatmap

# %%
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True, fmt=".2f", cmap="coolwarm", center=0,
    linewidths=0.5, ax=ax,
)
ax.set_title("Feature correlation matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Feature distributions by target class

# %%
target = "serious_dlqin2yrs"
feature_cols = df.select_dtypes("number").columns.drop(target)

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, feature_cols):
    sns.boxplot(data=df, x=target, y=col, ax=ax, showfliers=False)
    ax.set_title(col, fontsize=8)
    ax.set_xlabel("")
for ax in axes.flat[len(feature_cols):]:
    ax.set_visible(False)
fig.suptitle("Feature distributions by default (0=no, 1=yes)", fontsize=11)
fig.tight_layout()
plt.show()
