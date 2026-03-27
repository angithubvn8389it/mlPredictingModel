# Housing Price Prediction

This project trains a housing price prediction model from your CSV dataset.
It supports:

- Training and saving a model
- Evaluating model performance
- Generating predictions on new data

## Project Structure

```
mlPredictingModel/
  data/
    housing_price_dataset.csv
  src/housing_price_prediction/
    __init__.py
    config.py
    data.py
    features.py
    model.py
    train.py
    evaluate.py
    predict.py
  streamlit_app.py
  requirements.txt
  .gitignore
  README.md
```

## 1. Setup

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare Your Dataset

- File format: CSV
- Include one target column: `price`
- Use these feature columns for prediction:
  - `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`
  - `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`
  - `yr_built`, `yr_renovated`, `city`, `statezip`
- Columns not in the selected feature list (for example `street`, `country`, `date`) are ignored by the training/evaluation/prediction pipeline.

Example dataset path:

```text
data/housing_price_dataset.csv
```

## 3. Train the Model

**Windows (PowerShell)**
```powershell
$env:PYTHONPATH = "src"
python -m housing_price_prediction.train --data data/housing_price_dataset.csv
```

**macOS / Linux**
```bash
PYTHONPATH=src python -m housing_price_prediction.train --data data/housing_price_dataset.csv
```

Optional arguments:

- `--test-size` default `0.2`
- `--random-state` default `42`
- `--model-out` default `artifacts/model.joblib`
- `--metrics-out` default `artifacts/metrics.json`

## 4. Evaluate the Trained Model

**Windows (PowerShell)**
```powershell
$env:PYTHONPATH = "src"
python -m housing_price_prediction.evaluate --data data/housing_price_dataset.csv --model artifacts/model.joblib
```

**macOS / Linux**
```bash
PYTHONPATH=src python -m housing_price_prediction.evaluate --data data/housing_price_dataset.csv --model artifacts/model.joblib
```

## 5. Predict on New Data

Use a CSV that has the same feature columns as training data (target column is not required).

**Windows (PowerShell)**
```powershell
$env:PYTHONPATH = "src"
python -m housing_price_prediction.predict --data data/new_houses.csv --model artifacts/model.joblib --out artifacts/predictions.csv
```

**macOS / Linux**
```bash
PYTHONPATH=src python -m housing_price_prediction.predict --data data/new_houses.csv --model artifacts/model.joblib --out artifacts/predictions.csv
```

## Outputs

- Trained model: `artifacts/model.joblib`
- Training metrics: `artifacts/metrics.json`
- Predictions: `artifacts/predictions.csv`

## Notes

- If your dataset contains missing values, preprocessing handles them automatically.
- Categorical columns are one-hot encoded.
- Numeric columns are median-imputed and scaled.

## 6. Run Streamlit Web App

**Windows (PowerShell)**
```powershell
$env:PYTHONPATH = "src"
streamlit run streamlit_app.py
```

**macOS / Linux**
```bash
PYTHONPATH=src streamlit run streamlit_app.py
```

The web app supports:

- Single-house prediction with input controls
- Batch prediction from uploaded CSV
- Model information and metrics view
