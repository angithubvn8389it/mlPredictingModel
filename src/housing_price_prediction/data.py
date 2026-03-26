from pathlib import Path

import pandas as pd


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists() and not path.is_absolute() and path.parent == Path("."):
        fallback_path = Path("data") / path
        if fallback_path.exists():
            path = fallback_path

    if not path.exists():
        cwd = Path.cwd()
        raise FileNotFoundError(
            f"Dataset not found: {path}. Current working directory: {cwd}. "
            "Try using --data data/housing_price_dataset.csv"
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty")

    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
):
    if target_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Target column '{target_column}' does not exist. Available columns: {available}"
        )

    if feature_columns is None:
        X = df.drop(columns=[target_column])
    else:
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            available = ", ".join(df.columns)
            raise ValueError(
                "Missing feature columns in dataset: "
                f"{', '.join(missing_features)}. Available columns: {available}"
            )
        X = df[feature_columns].copy()

    y = df[target_column]
    return X, y
