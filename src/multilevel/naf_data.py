"""NAF 2025 hierarchy constants and data-loading utilities."""

import os

import numpy as np
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

# All 5 NAF 2025 levels — columns already present in naf_merged.parquet.
NACE_LEVELS: list[tuple[str, int | None]] = [
    ("sec", None),  # Section   — 21 codes (A–U)
    ("div", 2),     # Division  — 84 codes
    ("grp", 3),     # Group     — ~287 codes
    ("cls", 4),     # Class     — ~651 codes
    ("sub", 5),     # Sub-class — ~680 codes
]

_DEFAULT_URL = "https://minio.lab.sspcloud.fr/projet-text-classif/datasets/naf_merged.parquet"


def load_naf(cfg: dict, seed: int):
    """Load the merged NAF parquet (columns sec/div/grp/cls/sub already present)."""
    import pandas as pd

    url = os.environ.get("NAF_PARQUET_PATH", _DEFAULT_URL)
    level_names = [name for name, _ in NACE_LEVELS]
    df = pd.read_parquet(url, columns=["libelle_cleaned"] + level_names)
    df = df.dropna().reset_index(drop=True)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_test = cfg.get("test_size", 25000)
    n_val  = cfg.get("val_size",  25000)
    n      = len(df)

    df_test  = df.iloc[n - n_test:]
    df_rem   = df.iloc[:n - n_test]
    df_val   = df_rem.iloc[len(df_rem) - n_val:]
    df_train = df_rem.iloc[:len(df_rem) - n_val]

    fraction = cfg.get("train_fraction", 1.0)
    if fraction < 1.0:
        df_train = df_train.iloc[:max(1, int(len(df_train) * fraction))]

    encoders: dict[str, SKLabelEncoder] = {}
    for name, _ in NACE_LEVELS:
        le = SKLabelEncoder()
        le.fit(df_train[name].astype(str))
        encoders[name] = le

    def encode(split):
        X = split["libelle_cleaned"].values
        y = np.column_stack([
            encoders[name].transform(split[name].astype(str))
            for name, _ in NACE_LEVELS
        ])
        return X, y

    X_train, y_train = encode(df_train)
    X_val,   y_val   = encode(df_val)
    X_test,  y_test  = encode(df_test)

    num_classes_per_level = [len(encoders[name].classes_) for name, _ in NACE_LEVELS]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    for (name, _), nc in zip(NACE_LEVELS, num_classes_per_level):
        print(f"  Level '{name}': {nc} classes")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes_per_level
