from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from housing_price_prediction.config import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")
DATASET_PATH = Path("data/housing_price_dataset.csv")


@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@st.cache_data
def load_reference_dataset(dataset_path: Path):
    if not dataset_path.exists():
        return None
    return pd.read_csv(dataset_path)


def get_model_feature_columns(model) -> list[str]:
    _ = model
    # The runtime input schema is defined by configured raw feature columns.
    return FEATURE_COLUMNS.copy()


def validate_feature_columns(df: pd.DataFrame, expected_columns: list[str]):
    missing = [col for col in expected_columns if col not in df.columns]
    return missing


def predict_dataframe(model, df: pd.DataFrame, expected_columns: list[str]):
    features = df[expected_columns].copy()
    preds = model.predict(features)
    out = df.copy()
    out["predicted_price"] = preds
    return out


def build_single_input_row(expected_columns: list[str], reference_df: pd.DataFrame | None) -> pd.DataFrame:
    row_data = {}

    for col in expected_columns:
        series = None
        if reference_df is not None and col in reference_df.columns:
            series = reference_df[col]

        label = f"{col}"
        if series is not None and pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            default_value = float(clean_series.median()) if not clean_series.empty else 0.0
            is_integer = pd.api.types.is_integer_dtype(series)
            if is_integer:
                row_data[col] = int(
                    st.number_input(label, value=int(round(default_value)), step=1)
                )
            else:
                row_data[col] = float(st.number_input(label, value=default_value))
        else:
            default_text = ""
            unique_values = []
            if series is not None:
                clean_series = series.dropna().astype(str)
                if not clean_series.empty:
                    default_text = clean_series.iloc[0]
                    unique_values = sorted(clean_series.unique().tolist())

            if 1 < len(unique_values) <= 50:
                default_index = 0
                if default_text in unique_values:
                    default_index = unique_values.index(default_text)
                row_data[col] = st.selectbox(label, unique_values, index=default_index)
            else:
                row_data[col] = st.text_input(label, value=default_text)

    return pd.DataFrame([row_data])


def main():
    st.set_page_config(page_title="Housing Price Predictor", page_icon="house", layout="wide")
    st.title("Nền tảng dự đoán giá nhà")
    st.caption("Dự đoán giá nhà sử dụng mô hình hồi quy đã được huấn luyện")

    model = load_model(MODEL_PATH)
    metrics = load_metrics(METRICS_PATH)
    reference_df = load_reference_dataset(DATASET_PATH)

    if model is None:
        st.error(
            "Không tìm thấy mô hình tại artifacts/model.joblib. Hãy huấn luyện mô hình."
        )
        st.code(
            "$env:PYTHONPATH='src'; python -m housing_price_prediction.train --data data/housing_price_dataset.csv"
        )
        return

    model_feature_columns = get_model_feature_columns(model)
    target_column = TARGET_COLUMN
    if metrics and metrics.get("target"):
        target_column = str(metrics["target"])

    input_columns = FEATURE_COLUMNS.copy()

    if not input_columns:
        st.error("Không xác định được danh sách cột đặc trưng từ mô hình hoặc dataset.")
        return

    st.success("Mô hình đã chạy thành công!")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if metrics:
            st.metric("R2 (Coefficient)", f"{metrics.get('r2', 0):.4f}")
    with col_b:
        if metrics:
            st.metric("RMSE (Root Mean Square Error)", f"{metrics.get('rmse', 0):,.2f}")
    with col_c:
        if metrics:
            st.metric("MAE (Mean Absolute Error)", f"{metrics.get('mae', 0):,.2f}")

    tabs = st.tabs(["Ước lượng giá nhà", "Ước lượng giá nhà bằng file CSV", "Thông tin chi tiết về mô hình"])

    with tabs[0]:
        st.subheader("Dự đoán giá nhà")
        st.caption("Nhập theo đúng các cột của dataset (trừ cột mục tiêu price)")
        single_row_df = build_single_input_row(input_columns, reference_df)

        if st.button("Ước lượng giá nhà", type="primary"):
            prediction = model.predict(single_row_df[input_columns])[0]
            usd_to_vnd_rate = 26311.98
            prediction_vnd = prediction * usd_to_vnd_rate
            vnd_text = f"{prediction_vnd:,.0f}".replace(",", ".")
            st.success(f"Giá cả ước lượng: ${prediction:,.2f} (~ {vnd_text} VND)")

    with tabs[1]:
        st.subheader("Dự đoán theo dòng trong file CSV")
        st.write("Hãy tải file CSV có chứa các cột sau:")
        st.code(", ".join(input_columns))

        file = st.file_uploader("Tải file CSV lên", type=["csv"])
        if file is not None:
            batch_df = pd.read_csv(file)

            if target_column in batch_df.columns:
                batch_df = batch_df.drop(columns=[target_column])

            missing = validate_feature_columns(batch_df, input_columns)
            if missing:
                st.error(f"Thiếu các cột cần có: {', '.join(missing)}")
            else:
                predicted_df = predict_dataframe(model, batch_df, input_columns)
                st.success("Dự đoán theo dòng thành công!")
                st.dataframe(predicted_df.head(20), use_container_width=True)

                csv_data = predicted_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Tải file CSV dự đoán",
                    data=csv_data,
                    file_name="predictions_streamlit.csv",
                    mime="text/csv",
                )

    with tabs[2]:
        st.subheader("Chi tiết mô hình")
        st.write(f"Tập tin mô hình: {MODEL_PATH}")
        st.write(f"Loại mô hình: {type(model.named_steps['model']).__name__}")

        if metrics:
            st.write("Các chỉ số của mô hình:")
            st.json(metrics)
        else:
            st.info("File metrics.json chưa được tìm thấy. Hãy huấn luyện mô hình để tạo các chỉ số.")


if __name__ == "__main__":
    main()
