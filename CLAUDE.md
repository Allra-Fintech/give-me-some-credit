# CLAUDE.md

## Architecture

- **Logic** lives in `give_me_some_credit/` (the installable package). Never put business logic in notebooks.
- **Notebooks** in `notebooks/` are Jupytext percent-format `.py` files — they are the presentation layer only.
- **Papermill** executes notebooks programmatically; all notebooks must have a `parameters` tagged cell.
- `.ipynb` files are gitignored — only `.py` sources are committed.

## Working with notebooks

Convert `.py` → `.ipynb` before opening in Jupyter:

```bash
jupytext --to notebook notebooks/01_eda.py
```

Sync back after editing in Jupyter:

```bash
jupytext --to py:percent notebooks/01_eda.ipynb
```

## Package conventions

- All imports in notebooks go through the `give_me_some_credit` package.
- Column names are snake_case after loading (handled in `data/loader.py`).
- Feature engineering is stateless: `build_features(df) -> df`, no side effects.

## Adding a new notebook

1. Create `notebooks/NN_name.py` with the Jupytext header and a `# %% tags=["parameters"]` cell.
2. Include `output_path: str` in the parameters cell so Papermill can target it.

## Dependencies

Managed by Poetry. Add packages with `poetry add <pkg>`, not pip.
