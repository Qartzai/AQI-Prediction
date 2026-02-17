"""
Quick local predictions generator - trains a simple model and generates predictions
without needing to download from Hopsworks.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import requests
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import LAT, LON
from process_features import add_features

def fetch_forecast_data(days=3):
    """Fetch weather + AQ forecast"""
    print(f"Fetching {days}-day forecast...")
    
    aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&forecast_days={days}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
    wx_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m&forecast_days={days}"
    
    aq_data = requests.get(aq_url, timeout=30).json()["hourly"]
    wx_data = requests.get(wx_url, timeout=30).json()["hourly"]
    
    df_aq = pd.DataFrame(aq_data)
    df_wx = pd.DataFrame(wx_data)
    
    df = pd.merge(df_aq, df_wx, on="time", how="inner")
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    print(f"✅ Got {len(df)} hours of forecast")
    return df

def add_forecast_features(df):
    """Add features to forecast data WITHOUT computing AQI (we're predicting that!)"""
    df = df.copy()
    
    # Time features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    
    # Pollutant ratios
    df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
    
    # Meteorological combinations
    df["temp_humidity_ratio"] = df["temperature_2m"] / (df["relative_humidity_2m"] + 1e-6)
    df["wind_effect"] = df["wind_speed_10m"] * np.cos(np.deg2rad(df["wind_direction_10m"]))
    
    # AQI change rate - for forecast, use 0 (no historical AQI available)
    df["aqi_change_rate"] = 0
    
    return df

def quick_train_and_predict():
    """Train a quick model and generate predictions"""
    print("\n" + "="*60)
    print("QUICK PREDICTION GENERATOR")
    print("="*60 + "\n")
    
    # Load historical data
    print("1. Loading historical data...")
    df_hist = pd.read_csv("data/historical/historical_karachi_1y.csv")
    df_hist["datetime"] = pd.to_datetime(df_hist["datetime"])
    print(f"   ✅ Loaded {len(df_hist)} rows with features already computed")
    
    # Historical data already has features, no need to add them again
    
    # Drop leakage features
    print("2. Removing leakage features...")
    leakage = ["aqi_rolling_24h", "aqi_lag_1h", "high_pollution_flag"]
    df_hist.drop(columns=[c for c in leakage if c in df_hist.columns], errors="ignore", inplace=True)
    
    # Prepare training data
    feature_cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone", 
                    "sulphur_dioxide", "temperature_2m", "relative_humidity_2m", 
                    "wind_speed_10m", "hour", "day", "month", "weekday", 
                    "hour_sin", "aqi_change_rate", "pm_ratio", 
                    "temp_humidity_ratio", "wind_effect"]
    
    X = df_hist[feature_cols]
    y = df_hist["aqi"]
    
    # Quick train
    print("3. Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print("   ✅ Model trained")
    
    # Fetch forecast
    print("4. Fetching forecast data...")
    df_forecast = fetch_forecast_data(days=3)
    
    # Add features to forecast (WITHOUT calculating AQI - that's what we're predicting!)
    print("5. Preparing forecast features...")
    df_forecast = add_forecast_features(df_forecast)
    
    # Make predictions
    print("6. Generating predictions...")
    X_forecast = df_forecast[feature_cols]
    predictions = model.predict(X_forecast)
    
    # Create results
    results = pd.DataFrame({
        "datetime": df_forecast["datetime"],
        "predicted_aqi": predictions,
        "date": df_forecast["datetime"].dt.date
    })
    
    # Save
    os.makedirs("data/predictions", exist_ok=True)
    output_path = "data/predictions/next_3_days_predictions.csv"
    results.to_csv(output_path, index=False)
    
    print(f"\n✅ COMPLETE! Saved {len(results)} predictions to {output_path}")
    print(f"   Date range: {results['datetime'].min()} to {results['datetime'].max()}")
    print(f"   AQI range: {results['predicted_aqi'].min():.1f} to {results['predicted_aqi'].max():.1f}")
    print("="*60 + "\n")
    
    return results

if __name__ == "__main__":
    quick_train_and_predict()
