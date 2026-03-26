from dataclasses import dataclass
from pathlib import Path


TARGET_COLUMN = "price"
FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "city",
    "statezip",
]


@dataclass(frozen=True)
class TrainingConfig:
    target_column: str = TARGET_COLUMN
    test_size: float = 0.2
    random_state: int = 42
    model_output_path: Path = Path("artifacts/model.joblib")
    metrics_output_path: Path = Path("artifacts/metrics.json")


@dataclass(frozen=True)
class PredictionConfig:
    model_path: Path = Path("artifacts/model.joblib")
    output_path: Path = Path("artifacts/predictions.csv")
    prediction_column: str = "predicted_price"
