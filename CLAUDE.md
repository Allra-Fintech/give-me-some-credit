# CLAUDE.md

## Architecture

- **Logic** lives in `credit/` (the installable package). Never put business logic in notebooks.
- **Notebooks** in `notebooks/` are Jupytext percent-format `.py` files — they are the presentation layer only.
- **Papermill** executes notebooks programmatically; all notebooks must have a `parameters` tagged cell.
- `.ipynb` files are gitignored — only `.py` sources are committed.

## Working with notebooks

`jupytext` ships a JupyterLab extension — open `.py` notebooks directly without any conversion step:

```bash
jupyter lab notebooks/
```

Right-click a `.py` file → **Open With → Notebook**. Edits auto-save back to the `.py` source. No manual `jupytext` conversion needed.

## Package conventions

- All imports in notebooks go through the `credit` package.
- Column names are snake_case after loading (handled in `data/loader.py`); non-alphanumeric characters are also replaced with underscores.
- Feature engineering is stateless: `build_features(df) -> df`, no side effects.
- Model utilities in `credit.models`: `train`, `evaluate`, `compare_models` (5-fold CV AUC across all candidates), `calibration_curves`.
- Current model candidates: LogisticRegression, RandomForest, GradientBoosting, XGBoost.

## Data

Raw CSV files live in `data/` (`cs-training.csv`, `cs-test.csv`). Load via `credit.data.load_train()` — do not read files directly in notebooks.

## Adding a new notebook

1. Create `notebooks/NN_name.py` with the Jupytext header and a `# %% tags=["parameters"]` cell.
2. Include `output_path: str` in the parameters cell so Papermill can target it.

## Dependencies

Managed by Poetry. Add packages with `poetry add <pkg>`, not pip. XGBoost is a core (non-dev) dependency.
