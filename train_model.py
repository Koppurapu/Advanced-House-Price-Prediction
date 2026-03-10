from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_components import FeatureDropper, HouseDataPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_ROOT / "train.csv"
TEST_CSV = PROJECT_ROOT / "test.csv"
ARTIFACT_PATH = PROJECT_ROOT / "model_artifacts.joblib"
METRICS_PATH = PROJECT_ROOT / "metrics.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"
CURRENT_YEAR = datetime.now().year
RANDOM_STATE = 42
TARGET_COLUMN = "SalePrice"
IMPORTANT_COLUMNS = [
    "SalePrice",
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "Neighborhood",
    "LotArea",
    "KitchenQual",
]


@dataclass
class ModelResult:
    name: str
    rmse: float
    r2: float
    pipeline: Pipeline


def build_pipeline(add_engineering: bool) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    feature_processor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipe, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("domain_preprocess", HouseDataPreprocessor(add_engineering=add_engineering)),
            ("drop_unused", FeatureDropper(columns_to_drop=["Id"])),
            ("feature_processor", feature_processor),
            ("model", LinearRegression()),
        ]
    )


def evaluate_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, name: str) -> ModelResult:
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)
    rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))
    r2 = float(r2_score(y_valid, preds))
    return ModelResult(name=name, rmse=rmse, r2=r2, pipeline=pipeline)


def build_metadata(feature_df: pd.DataFrame) -> dict[str, Any]:

    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=np.number).columns.tolist()

    numeric_defaults = {col: float(feature_df[col].median()) if not feature_df[col].dropna().empty else 0.0 for col in numeric_cols}
    categorical_defaults: dict[str, str] = {}
    categorical_options: dict[str, list[str]] = {}

    for col in categorical_cols:
        col_modes = feature_df[col].mode(dropna=True)
        default_value = str(col_modes.iloc[0]) if not col_modes.empty else "None"
        values = feature_df[col].dropna().astype(str).unique().tolist()
        options = sorted(set(values + [default_value, "None"]))
        categorical_defaults[col] = default_value
        categorical_options[col] = options

    return {
        "feature_order": feature_df.columns.tolist(),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
        "categorical_options": categorical_options,
        "current_year": CURRENT_YEAR,
    }


def main() -> None:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    feature_columns = [col for col in IMPORTANT_COLUMNS if col != TARGET_COLUMN]
    required_train_cols = [TARGET_COLUMN, *feature_columns]

    missing_train_cols = [col for col in required_train_cols if col not in train_df.columns]
    if missing_train_cols:
        raise ValueError(f"Missing required columns in train.csv: {missing_train_cols}")

    missing_test_cols = [col for col in feature_columns if col not in test_df.columns]
    if missing_test_cols:
        raise ValueError(f"Missing required columns in test.csv: {missing_test_cols}")

    X = train_df[feature_columns].copy()
    y = train_df[TARGET_COLUMN]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    baseline_pipeline = build_pipeline(add_engineering=False)
    engineered_pipeline = build_pipeline(add_engineering=True)

    baseline_result = evaluate_model(
        baseline_pipeline, X_train, y_train, X_valid, y_valid, "LinearRegression_Baseline"
    )
    engineered_result = evaluate_model(
        engineered_pipeline, X_train, y_train, X_valid, y_valid, "LinearRegression_Engineered"
    )

    results = [baseline_result, engineered_result]
    best_result = min(results, key=lambda r: r.rmse)

    best_result.pipeline.fit(X, y)
    test_predictions = best_result.pipeline.predict(test_df[feature_columns].copy())

    metrics_df = pd.DataFrame(
        {
            "Model": [r.name for r in results],
            "RMSE": [r.rmse for r in results],
            "R2": [r.r2 for r in results],
        }
    ).sort_values(by="RMSE")
    metrics_df.to_csv(METRICS_PATH, index=False)

    submission_df = pd.DataFrame({"Id": test_df["Id"], "SalePrice": test_predictions})
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    metadata = build_metadata(X)
    artifacts = {
        "best_model_name": best_result.name,
        "pipeline": best_result.pipeline,
        "metrics": metrics_df.to_dict(orient="records"),
        "metadata": metadata,
    }
    joblib.dump(artifacts, ARTIFACT_PATH)

    print("Training complete")
    print(f"Best model: {best_result.name}")
    print(f"Saved artifacts: {ARTIFACT_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved submission: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
