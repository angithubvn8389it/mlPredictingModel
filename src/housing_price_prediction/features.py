import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Fixed reference year consistent with the King County 2014-2015 dataset period.
# Using a constant keeps predictions stable across train and inference.
_REFERENCE_YEAR = 2015


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived numeric features to improve model accuracy.

    All new columns are numeric, so CatBoost treats them as continuous
    features without any additional configuration.

    New features
    ------------
    house_age               : reference year minus yr_built
    is_renovated            : 1 if yr_renovated > 0, else 0
    years_since_renovation  : years since last renovation (or since built)
    sqft_per_bedroom        : sqft_living / bedrooms (clamped ≥ 1)
    living_lot_ratio        : sqft_living / sqft_lot (clamped ≥ 1)
    basement_ratio          : sqft_basement / sqft_living (clamped ≥ 1)
    bath_bed_ratio          : bathrooms / bedrooms (clamped ≥ 1)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if "yr_built" in df.columns:
            df["house_age"] = (_REFERENCE_YEAR - df["yr_built"]).clip(lower=0)

        if "yr_renovated" in df.columns and "yr_built" in df.columns:
            renovated = df["yr_renovated"] > 0
            df["is_renovated"] = renovated.astype(int)
            base_age = (_REFERENCE_YEAR - df["yr_built"]).clip(lower=0)
            df["years_since_renovation"] = np.where(
                renovated,
                (_REFERENCE_YEAR - df["yr_renovated"]).clip(lower=0),
                base_age,
            )

        if "sqft_living" in df.columns and "bedrooms" in df.columns:
            df["sqft_per_bedroom"] = df["sqft_living"] / df["bedrooms"].clip(lower=1)

        if "sqft_living" in df.columns and "sqft_lot" in df.columns:
            df["living_lot_ratio"] = df["sqft_living"] / df["sqft_lot"].clip(lower=1)

        if "sqft_basement" in df.columns and "sqft_living" in df.columns:
            df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"].clip(lower=1)

        if "bathrooms" in df.columns and "bedrooms" in df.columns:
            df["bath_bed_ratio"] = df["bathrooms"] / df["bedrooms"].clip(lower=1)

        return df
