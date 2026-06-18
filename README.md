# Predictive Analytics Using Historical Data

A comprehensive data science pipeline built to transform raw historical time-series observations into future trend forecasts. This project demonstrates end-to-end predictive modeling workflows, data cleaning, lag feature engineering, and statistical validation.

## 🚀 Key Features

- **Automated Data Preprocessing**: Robust cleaning pipeline that handles missing gaps using forward-fill (`.ffill()`) imputation and caps extreme outliers using a 99th percentile threshold.
- **Temporal Feature Engineering**: Generates calendar features (Year, Month, Day, Day of Week) and rolling window lag metrics (`Sales_Lag_1`, `Sales_Lag_7`, `Rolling_Mean_7`) to capture historical sequential dependencies.
- **Predictive Machine Learning Regressor**: Deploys an ensemble `RandomForestRegressor` to map multi-variable interactions and underlying cyclical weekly seasonality.
- **Performance Evaluation & Visualization**: Measures predictive precision out-of-sample using standard statistical metrics (MAE, RMSE, $R^2$) and renders multi-layered trend validation plots.

## 📊 Model Performance Metrics

- **Mean Absolute Error (MAE)**: 24.41
- **Root Mean Squared Error (RMSE)**: 29.42
- **R-squared Score ($R^2$)**: 0.39

### 🔍 Core Takeaway: The Extrapolation Constraint

This implementation highlights a foundational machine learning concept: tree-based ensemble regressors excel at capturing complex, non-linear cyclical seasonal waves (the weekly zig-zag patterns), but natively struggle to extrapolate upward linear trends beyond the maximum boundaries seen in the training dataset.

## 📁 Repository Structure

- `predictive_model.py` -> Core Python script containing data generation, cleaning, training, and evaluation logic.
- `.gitignore` -> Prevents temporary Python cache files (`__pycache__/`) from entering the repository.
- `README.md` -> Project documentation and analysis overview.

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

## 🏃 How to Execute the Pipeline

1. Clone this repository to your local machine.
2. Install the required calculation and visualization libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
3. Run the training and forecasting script:
   ```bash
   python predictive_model.py
   ```
