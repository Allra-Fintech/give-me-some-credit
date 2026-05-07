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
```

## Setup (macOS / Linux)

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

## Setup (Windows)

Install Poetry via PowerShell:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

If PowerShell blocks script execution, allow it first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then install dependencies and activate the environment:

```powershell
poetry install

# activate
& $(poetry env info --path)\Scripts\Activate.ps1

# or run one-off commands without activating
poetry run python ...
```

Install the Jupyter kernel:

```powershell
python -m ipykernel install --user --name give-me-some-credit
```

## Interactive development (JupyterLab)

`jupytext` ships with a JupyterLab extension — open `.py` notebooks directly as notebooks, write code, and run cells without any conversion step:

```bash
jupyter lab notebooks/
```

Right-click a `.py` file → **Open With → Notebook**. Edits are saved back to the `.py` file automatically.

## Automated / parameterized runs (Papermill)

Use Papermill for batch execution with overridden parameters (e.g. in CI or experiments). It requires a `.ipynb` input, so convert first:

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
