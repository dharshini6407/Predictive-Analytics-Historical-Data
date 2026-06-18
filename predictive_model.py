import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------
# 1. GENERATE SYNTHETIC HISTORICAL DATA (Simulating real-world data)
# -------------------------------------------------------------
print("Generating synthetic historical sales data...")
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
n_days = len(dates)

# Build components: Baseline + upward trend + weekly seasonality + noise
baseline = 200
trend = np.linspace(0, 150, n_days)
seasonality = 50 * np.sin(2 * np.pi * dates.dayofweek / 7)
noise = np.random.normal(0, 20, n_days)
sales = baseline + trend + seasonality + noise

# Create a DataFrame containing intentional missing values and outliers
df = pd.DataFrame({"Date": dates, "Sales": sales})
df.loc[df.sample(frac=0.02, random_state=42).index, "Sales"] = np.nan  # Missing data
df.loc[df.sample(n=5, random_state=24).index, "Sales"] = 9999          # Anomalous outliers

# -------------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# -------------------------------------------------------------
print("Cleaning and preprocessing data...")
# Extract calendar features from date
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Handle Outliers: Cap extreme values using the 99th percentile threshold
upper_threshold = df["Sales"].quantile(0.99)
df.loc[df["Sales"] > upper_threshold, "Sales"] = np.nan

# Handle Missing Values: Use forward-fill to impute based on the previous day's sales
df["Sales"] = df["Sales"].ffill()

# Create rolling features to capture recent trends (lag features)
df["Sales_Lag_1"] = df["Sales"].shift(1)
df["Sales_Lag_7"] = df["Sales"].shift(7)
df["Rolling_Mean_7"] = df["Sales"].shift(1).rolling(window=7).mean()

# Drop rows created with NaN values from shifting/rolling windows
df = df.dropna().reset_index(drop=True)

# -------------------------------------------------------------
# 3. SPLIT DATA INTO TRAIN AND TEST SETS
# -------------------------------------------------------------
# Define predictors (X) and target variable (y)
features = ["Year", "Month", "Day", "DayOfWeek", "Sales_Lag_1", "Sales_Lag_7", "Rolling_Mean_7"]
X = df[features]
y = df["Sales"]

# Split data sequentially to preserve temporal ordering
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
test_dates = df.loc[X_test.index, "Date"]

# -------------------------------------------------------------
# 4. TRAIN THE PREDICTIVE MODEL
# -------------------------------------------------------------
print("Training Random Forest Regressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------------------
# 5. EVALUATE MODEL ACCURACY
# -------------------------------------------------------------
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n=== Model Evaluation Metrics ===")
print(f"Mean Absolute Error (MAE):    {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score (R²):          {r2:.2f}")

# -------------------------------------------------------------
# 6. VISUALIZE PREDICTIONS
# -------------------------------------------------------------
print("\nGenerating visualization...")
plt.figure(figsize=(14, 7))
plt.plot(df["Date"], df["Sales"], label="Historical Actual Sales", color="blue", alpha=0.4)
plt.plot(test_dates, y_test, label="True Out-of-Sample Sales", color="black", lw=1.5)
plt.plot(test_dates, predictions, label="Model Forecasted Trends", color="red", linestyle="--", lw=1.5)

plt.title("Predictive Analytics: Historical Sales vs Future Forecasted Trends", fontsize=14)
plt.xlabel("Timeline", fontsize=12)
plt.ylabel("Sales Units", fontsize=12)
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()
