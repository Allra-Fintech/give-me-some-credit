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
```

Activate the virtual environment (Poetry 2.0+):

```bash
# Option 1 — recommended
source $(poetry env info --path)/bin/activate

# Option 2 — one-off commands without activating
poetry run python ...
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

## Watch notebooks (auto-convert on save)

`watchdog` (installed as a dev dependency) can re-run Jupytext whenever a `.py` notebook changes:

```bash
watchmedo shell-command \
  --patterns="*.py" \
  --recursive \
  --command='jupytext --to notebook "${watch_src_path}"' \
  notebooks/
```

## Run a notebook with Papermill

Papermill requires a `.ipynb` input, so convert with Jupytext first:

```bash
jupytext --to notebook notebooks/03_model_training.py

papermill notebooks/03_model_training.ipynb outputs/03_model_training.ipynb \
  -p n_estimators 200 \
  -p max_depth 5
```

## Data

Place the Kaggle files in `data/`:

- `cs-training.csv`
- `cs-test.csv`

Download from: https://www.kaggle.com/competitions/GiveMeSomeCredit/data
