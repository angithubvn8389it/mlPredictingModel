import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from housing_price_prediction.config import FEATURE_COLUMNS, TARGET_COLUMN
from housing_price_prediction.data import load_dataset, split_features_target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained housing price model")
    parser.add_argument("--data", required=True, help="Path to evaluation CSV dataset")
    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help="Target column name in dataset",
    )
    parser.add_argument(
        "--model",
        default="artifacts/model.joblib",
        help="Path to trained model file",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    model = joblib.load(args.model)
    df = load_dataset(args.data)
    X, y = split_features_target(df, args.target, FEATURE_COLUMNS)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    metrics = {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y, preds)),
        "rows": int(len(df)),
    }

    print("Evaluation metrics")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
