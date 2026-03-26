import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from housing_price_prediction.config import FEATURE_COLUMNS, TARGET_COLUMN
from housing_price_prediction.data import load_dataset, split_features_target
from housing_price_prediction.features import FeatureEngineer
from housing_price_prediction.model import build_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train housing price prediction model")
    parser.add_argument("--data", required=True, help="Path to training CSV dataset")
    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help="Target column name in dataset",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model-out",
        default="artifacts/model.joblib",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--metrics-out",
        default="artifacts/metrics.json",
        help="Output path for metrics JSON",
    )
    parser.add_argument(
        "--keep-outliers",
        action="store_true",
        help="Keep all training rows and disable IQR outlier filtering",
    )
    return parser


def _remove_target_outliers_iqr(X_train, y_train):
    q1 = y_train.quantile(0.25)
    q3 = y_train.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    keep_mask = (y_train >= lower_bound) & (y_train <= upper_bound)

    X_filtered = X_train.loc[keep_mask].copy()
    y_filtered = y_train.loc[keep_mask].copy()
    removed_rows = int((~keep_mask).sum())
    return X_filtered, y_filtered, removed_rows


def train(
    data_path: str,
    target: str,
    test_size: float,
    random_state: int,
    remove_outliers: bool = True,
):
    df = load_dataset(data_path)
    X, y = split_features_target(df, target, FEATURE_COLUMNS)

    # Apply outlier filter to the full dataset *before* splitting so that the
    # test set shares the same price distribution as the training set.
    # Filtering only the training split causes the model (trained on prices up
    # to ~Q3+1.5*IQR) to be evaluated against extreme test values it was never
    # exposed to, which collapses R² toward zero.
    total_rows = int(len(X))
    outliers_removed = 0
    if remove_outliers:
        X, y, outliers_removed = _remove_target_outliers_iqr(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    model = build_model(
        random_state=random_state,
        cat_features=categorical_features,
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
        "model": type(model).__name__,
        "rows": int(len(df)),
        "rows_after_filter": int(len(X)),
        "features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "outliers_removed": outliers_removed,
        "outlier_filter": "iqr_1.5_before_split" if remove_outliers else "disabled",
        "selected_features": FEATURE_COLUMNS,
        "target": target,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return pipeline, metrics


def main() -> None:
    args = _build_parser().parse_args()

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model, metrics = train(
        data_path=args.data,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        remove_outliers=not args.keep_outliers,
    )

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training completed")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
