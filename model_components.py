from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


CURRENT_YEAR = datetime.now().year


class HouseDataPreprocessor(BaseEstimator, TransformerMixin):
    """Domain-aware missing value handling and optional feature engineering."""

    domain_none_cols = [
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "MasVnrType",
    ]

    domain_zero_cols = [
        "MasVnrArea",
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
    ]

    def __init__(self, add_engineering: bool = True):
        self.add_engineering = add_engineering

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HouseDataPreprocessor":
        df = X.copy()
        self.columns_ = list(df.columns)

        self.median_by_neighborhood_ = None
        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            self.median_by_neighborhood_ = (
                df.groupby("Neighborhood", dropna=False)["LotFrontage"].median().to_dict()
            )

        self.numeric_fill_values_: dict[str, float] = {}
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                self.numeric_fill_values_[col] = 0.0
                continue

            # Use median for skewed numeric distributions and mean otherwise.
            strategy_value = float(series.median()) if abs(series.skew()) > 1 else float(series.mean())
            self.numeric_fill_values_[col] = strategy_value

        self.categorical_fill_values_: dict[str, str] = {}
        cat_cols = df.select_dtypes(exclude=np.number).columns
        for col in cat_cols:
            modes = df[col].mode(dropna=True)
            self.categorical_fill_values_[col] = str(modes.iloc[0]) if not modes.empty else "None"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col in self.domain_none_cols:
            if col in df.columns:
                df[col] = df[col].fillna("None")

        for col in self.domain_zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if "LotFrontage" in df.columns and self.median_by_neighborhood_ is not None:
            if "Neighborhood" in df.columns:
                df["LotFrontage"] = df.apply(
                    lambda row: self.median_by_neighborhood_.get(row["Neighborhood"], np.nan)
                    if pd.isna(row["LotFrontage"])
                    else row["LotFrontage"],
                    axis=1,
                )
            df["LotFrontage"] = df["LotFrontage"].fillna(self.numeric_fill_values_.get("LotFrontage", 0.0))

        for col, value in self.numeric_fill_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        for col, value in self.categorical_fill_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        if self.add_engineering:
            if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
                df["TotalArea"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
            if "YearBuilt" in df.columns:
                df["HouseAge"] = CURRENT_YEAR - df["YearBuilt"]

        return df


class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop: list[str]):
        self.columns_to_drop = columns_to_drop

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.columns_to_drop if c in X.columns], errors="ignore")