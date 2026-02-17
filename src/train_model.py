# Purpose: Train and evaluate 3 models (Ridge, RF, XGBoost)

import hopsworks
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load API Key and Connect to Hopsworks
api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")

try:
    # First try the local historical data with 360 days
    hist_path = os.path.join(os.path.dirname(__file__), "../data/historical/historical_karachi_1y.csv")
    if os.path.exists(hist_path):
        df = pd.read_csv(hist_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        print(f"✅ Loaded 360 days from {hist_path}")
        print(f"Initial shape: {df.shape}")
    else:
        # If no local file, try Hopsworks
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store(name='aqi_predictor_qartzai_featurestore')
        print("✅ Connected to Hopsworks Feature Store")

        fg = fs.get_feature_group("qartzai_2", version=1)
        df = fg.read()
        
        # Handle datetime - make timezone-naive
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            if df["datetime"].dt.tz is not None:
                df["datetime"] = df["datetime"].dt.tz_localize(None)
        
        print("📥 Data fetched from Hopsworks successfully!")
        print(f"Initial shape: {df.shape}")
except Exception as e:
    print("⚠️ Could not load data:", str(e))
    print("📁 Using fallback file...")
    df = pd.read_csv("../data/final/final_selected_features.csv")
    print(f"Initial shape: {df.shape}")

print("Initial shape:", df.shape)

# 2. Prepare datetime and ensure correct type
if "datetime_str" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime_str"])
    df.drop(columns=["datetime_str"], inplace=True)

# 3. Sort chronologically for time-based split
df = df.sort_values(by="datetime").reset_index(drop=True)

# 4. Drop high-leakage AQI features
# Features based on AQI or future information
leakage_features = [
    "aqi_rolling_24h",  # Rolling average of AQI (leakage)
    "aqi_lag_1h",       # Lagged AQI values (leakage)
    "high_pollution_flag"  # Flag based on AQI itself (leakage)
]
for col in leakage_features:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"⚠️ Dropped leakage feature: {col}")

# 5. Add ±5% random noise to pollutant readings (simulate sensor variability)
np.random.seed(42)
pollutant_cols = ["pm10", "pm2_5", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]
for col in pollutant_cols:
    if col in df.columns:
        df[col] = df[col] * (1 + np.random.normal(0, 0.05, len(df)))
print("🌫️ Added ±5% Gaussian noise to pollutant columns for realistic variation")

# 6. Remove duplicates & check missing values
df = df.drop_duplicates().reset_index(drop=True)
print("\n🔍 Missing values after cleaning:")
print(df.isna().sum())

# 7. Time-based split to prevent leakage
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Drop datetime from model features (after split) 
X_train = train_df.drop(columns=["aqi", "datetime"])
y_train = train_df["aqi"]
X_test = test_df.drop(columns=["aqi", "datetime"])
y_test = test_df["aqi"]

print(f"✅ Time-based split complete → Train: {X_train.shape}, Test: {X_test.shape}")
print(f"📋 Training features ({len(X_train.columns)}): {sorted(X_train.columns.tolist())}")

# 8. Preprocessing (Scaling for Ridge only) 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Model Training 
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )
}

results = {}

print("\n🚀 Training Models...\n")
for name, model in models.items():
    if name == "Ridge Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
    print(f"✅ {name} → RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# 10. Compare Results
results_df = pd.DataFrame(results).T.sort_values(by="RMSE")
print("\n📊 Model Comparison:\n")
print(results_df)

# 11. Save Best Model (with safety checks & confirmation) 
from joblib import dump

best_model_name = results_df.index[0].strip()
print(f"\n🏆 Best Model Selected: {best_model_name}")

# Ensure the models directory exists
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
os.makedirs(model_dir, exist_ok=True)

# Define file paths
model_path = os.path.join(model_dir, f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

print(f"📁 Model will be saved at: {model_path}")

# Try saving model locally
try:
    dump(models[best_model_name], model_path)
    if best_model_name == "Ridge Regression":
        dump(scaler, scaler_path)
        print(f"💾 Scaler also saved → {scaler_path}")
    print(f"✅ Model saved locally at {model_path}")
except Exception as e:
    print("⚠️ Error saving model locally:", e)

# Upload model to Hopsworks Model Registry
print("\n📤 Uploading model to Hopsworks Model Registry...")
try:
    if api_key:
        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()
        
        # Create model metadata with proper schema types
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema
        
        # Create input schema with types
        input_features = [{"name": col, "type": "double"} for col in X_train.columns]
        output_features = [{"name": "aqi", "type": "double"}]
        
        input_schema = Schema(input_features)
        output_schema = Schema(output_features)
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        
        # Register model (metrics must be numeric only)
        aqi_model = mr.python.create_model(
            name="aqi_prediction_model",
            version=3,  # Explicitly set version
            metrics={
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            },
            model_schema=model_schema,
            description=f"AQI prediction model using {best_model_name}. Trained on {len(X_train)} samples.",
        )
        
        # Save model files
        aqi_model.save(model_path)
        if best_model_name == "Ridge Regression" and os.path.exists(scaler_path):
            # Save scaler alongside model
            import shutil
            temp_dir = os.path.join(model_dir, "temp_model_dir")
            os.makedirs(temp_dir, exist_ok=True)
            shutil.copy(model_path, temp_dir)
            shutil.copy(scaler_path, temp_dir)
            aqi_model.save(temp_dir)
            shutil.rmtree(temp_dir)
        
        print(f"✅ Model uploaded to Hopsworks Model Registry")
    else:
        print("⚠️ Skipping Hopsworks upload (no API key)")
except Exception as e:
    print(f"⚠️ Could not upload to Hopsworks Model Registry: {e}")
    print("   Model is still saved locally")

# --- Extra Info ---
print("\n📊 Model Performance Summary ---")
print(results_df)
print("\nAQI range:", y_train.min(), "to", y_train.max())
print("Test RMSE % of range:", (results_df.iloc[0]['RMSE'] / (y_train.max() - y_train.min())) * 100)

train_preds = models["Random Forest"].predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
print(f"Train RMSE: {train_rmse:.3f}, Test RMSE: {rmse:.3f}")