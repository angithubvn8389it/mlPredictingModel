import argparse
from pathlib import Path

import joblib

from housing_price_prediction.config import FEATURE_COLUMNS
from housing_price_prediction.data import load_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run housing price prediction")
    parser.add_argument("--data", required=True, help="Path to input CSV dataset")
    parser.add_argument(
        "--model",
        default="artifacts/model.joblib",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--out",
        default="artifacts/predictions.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--prediction-col",
        default="predicted_price",
        help="Prediction column name",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    model = joblib.load(args.model)
    df = load_dataset(args.data)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            "Missing feature columns in input data: "
            f"{', '.join(missing_features)}"
        )

    feature_df = df[FEATURE_COLUMNS].copy()

    predictions = model.predict(feature_df)
    result = df.copy()
    result[args.prediction_col] = predictions

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
