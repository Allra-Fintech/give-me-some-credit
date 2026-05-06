from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "cs-training.csv", index_col=0)
    df.columns = [_snake(c) for c in df.columns]
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "cs-test.csv", index_col=0)
    df.columns = [_snake(c) for c in df.columns]
    return df


def _snake(name: str) -> str:
    import re
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()
