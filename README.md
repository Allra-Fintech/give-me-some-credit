# Give Me Some Credit

Credit default risk prediction pipeline — Kaggle competition dataset.

## Architecture

```
Notebook (.py)  →  presentation layer (Jupytext)
Python package  →  logic (data, features, models)
Papermill       →  parameterized execution & automation
```

## Project layout

```
credit/   # installable package — all logic lives here
  data/                #   load_train(), load_test()
  features/            #   build_features()
  models/              #   train(), evaluate()
  utils/               #   get_logger()
notebooks/             # Jupytext percent-format .py files (version-control friendly)
  01_eda.py
  02_feature_engineering.py
  03_model_training.py
  04_evaluation.py
outputs/               # Papermill-executed .ipynb outputs (gitignored except .gitkeep)
data/                  # raw CSVs (gitignored)
tests/
```

## Setup

```bash
poetry install
poetry shell
```

Install the Jupyter kernel for this env:

```bash
python -m ipykernel install --user --name give-me-some-credit
```

Convert a notebook `.py` → `.ipynb` to open in Jupyter:

```bash
jupytext --to notebook notebooks/01_eda.py
```

Or configure Jupytext to sync automatically — edit `~/.jupyter/jupyter_notebook_config.py`:

```python
c.ContentsManager.default_jupytext_formats = "ipynb,py:percent"
```

## Run a notebook with Papermill

```bash
papermill notebooks/03_model_training.py outputs/03_model_training.ipynb \
  -p n_estimators 200 \
  -p max_depth 5
```

## Data

Place the Kaggle files in `data/`:

- `cs-training.csv`
- `cs-test.csv`

Download from: https://www.kaggle.com/competitions/GiveMeSomeCredit/data
