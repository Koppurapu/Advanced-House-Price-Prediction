# House Prices - Linear Regression with Feature Engineering

This project trains a house price prediction model using the Kaggle House Prices dataset and deploys it with Streamlit.

## What this project does

- Handles missing values with mixed strategies:
  - Domain logic (fill `None` or `0` for features where missing has semantic meaning)
  - Neighborhood median for `LotFrontage`
  - Mean/median for numeric columns depending on skewness
  - Mode for categorical columns
- Adds required engineered features:
  - `TotalArea = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
  - `HouseAge = current_year - YearBuilt`
- Trains and compares two Linear Regression variants:
  - Baseline (without engineered features)
  - Engineered (with engineered features)
- Compares models using:
  - RMSE
  - R²
- Saves best model artifacts and generates predictions for `test.csv`.
- Serves predictions through a Streamlit app with a dynamic input form.

## Files generated after training

- `model_artifacts.joblib` - Best model pipeline + metadata + metrics
- `metrics.csv` - RMSE and R² comparison table
- `submission.csv` - Predictions for Kaggle `test.csv`

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Notes

- The Streamlit input form is generated from trained metadata, so it adapts directly to the model feature schema.
- If you retrain the model, the app will automatically use the latest saved artifacts.
