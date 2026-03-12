
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the superconductivity dataset."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place the UCI CSV at data/superconductor.csv."
        )
    return pd.read_csv(csv_path)


def split_features_target(
    data: pd.DataFrame,
    target_col: str = "critical_temp",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features and target."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return X, y


def train_test_data(
    data: pd.DataFrame,
    target_col: str = "critical_temp",
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
):
    """Return train/test splits for features and target."""
    X, y = split_features_target(data, target_col=target_col)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit a scaler on the training data and transform both train and test sets."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler
