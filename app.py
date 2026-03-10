from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = PROJECT_ROOT / "model_artifacts.joblib"


@st.cache_resource
def load_artifacts() -> dict:
    return joblib.load(ARTIFACT_PATH)


def build_input_frame(metadata: dict) -> pd.DataFrame:
    st.subheader("Enter House Details")
    st.caption("The form fields are generated dynamically from the trained model metadata.")

    numeric_features: list[str] = metadata["numeric_features"]
    categorical_features: list[str] = metadata["categorical_features"]
    feature_order: list[str] = metadata["feature_order"]

    numeric_defaults: dict[str, float] = metadata["numeric_defaults"]
    categorical_defaults: dict[str, str] = metadata["categorical_defaults"]
    categorical_options: dict[str, list[str]] = metadata["categorical_options"]

    values: dict[str, object] = {}

    with st.form("predict_form"):
        left_col, right_col = st.columns(2)

        # Numeric inputs
        for idx, feature in enumerate(numeric_features):
            col = left_col if idx % 2 == 0 else right_col
            default_val = float(numeric_defaults.get(feature, 0.0))
            values[feature] = col.number_input(
                label=feature,
                value=default_val,
                step=1.0,
                format="%.4f",
            )

        # Categorical inputs
        for idx, feature in enumerate(categorical_features):
            col = left_col if idx % 2 == 0 else right_col
            options = categorical_options.get(feature, ["None"])
            default = categorical_defaults.get(feature, options[0])
            default_index = options.index(default) if default in options else 0
            values[feature] = col.selectbox(
                label=feature,
                options=options,
                index=default_index,
            )

        submitted = st.form_submit_button("Predict Sale Price")

    if not submitted:
        st.stop()

    ordered_values = {feature: values.get(feature) for feature in feature_order}
    return pd.DataFrame([ordered_values])


def main() -> None:
    st.set_page_config(page_title="House Price Predictor", layout="wide")
    st.title("House Sale Price Predictor")
    st.write("Linear Regression model trained with domain-aware preprocessing and feature engineering.")

    if not ARTIFACT_PATH.exists():
        st.error("Model artifacts not found. Run train_model.py first.")
        st.stop()

    artifacts = load_artifacts()
    pipeline = artifacts["pipeline"]
    metadata = artifacts["metadata"]
    metrics = artifacts["metrics"]
    best_model_name = artifacts["best_model_name"]

    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)
    st.success(f"Best model by RMSE: {best_model_name}")

    input_df = build_input_frame(metadata)
    prediction = float(pipeline.predict(input_df)[0])

    st.subheader("Predicted Price")
    st.metric("Estimated Sale Price", f"${prediction:,.2f}")


if __name__ == "__main__":
    main()
