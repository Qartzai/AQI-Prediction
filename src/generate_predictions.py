"""
Prediction Pipeline: Generate future AQI predictions using weather forecasts
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import load
import requests

try:
    from src.config import LAT, LON, SAVE_LOCAL
    from src.process_features import add_features
    from src.clean_data import clean_data
    from src.process_data import process_latest_json
except ModuleNotFoundError:
    from config import LAT, LON, SAVE_LOCAL
    from process_features import add_features
    from clean_data import clean_data
    from process_data import process_latest_json


def fetch_weather_forecast(forecast_days=3):
    """Fetch weather + air quality forecast from Open-Meteo"""
    print(f"🌤️ Fetching {forecast_days}-day weather forecast...")
    
    # Air quality forecast
    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&forecast_days={forecast_days}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
    )
    
    # Weather forecast
    wx_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
        f"&forecast_days={forecast_days}"
    )
    
    try:
        aq_response = requests.get(aq_url, timeout=30)
        wx_response = requests.get(wx_url, timeout=30)
        
        aq_response.raise_for_status()
        wx_response.raise_for_status()
        
        aq_data = aq_response.json().get("hourly", {})
        wx_data = wx_response.json().get("hourly", {})
        
        df_aq = pd.DataFrame(aq_data)
        df_wx = pd.DataFrame(wx_data)
        
        # Merge on time
        df = pd.merge(df_aq, df_wx, on="time", how="inner")
        df.rename(columns={"time": "datetime"}, inplace=True)
        
        print(f"✅ Fetched {len(df)} hours of forecast data")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching forecast: {e}")
        return None


def generate_predictions(forecast_days=3):
    """Complete prediction pipeline"""
    print("\n🚀 Starting Prediction Pipeline\n")
    print("=" * 50)
    
    # 1. Load trained model
    print("\n1️⃣ Loading trained model...")
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    
    # Find the best model file
    model_files = [f for f in os.listdir(model_dir) if f.startswith("best_model_") and f.endswith(".pkl")]
    
    if not model_files:
        print("❌ No trained model found! Please run train_model.py first.")
        return None
    
    model_path = os.path.join(model_dir, model_files[0])
    model = load(model_path)
    print(f"✅ Loaded model: {model_files[0]}")
    
    # Load scaler if exists (for Ridge regression)
    scaler = None
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = load(scaler_path)
        print(f"✅ Loaded scaler: scaler.pkl")
    
    # 2. Fetch forecast data
    print(f"\n2️⃣ Fetching {forecast_days}-day weather forecast...")
    forecast_df = fetch_weather_forecast(forecast_days=forecast_days)
    
    if forecast_df is None or forecast_df.empty:
        print("❌ Failed to fetch forecast data")
        return None
    
    # 3. Process forecast data (same pipeline as training)
    print("\n3️⃣ Processing forecast data...")
    processed_df = process_latest_json(forecast_df)
    cleaned_df = clean_data(processed_df)
    
    # For forecasts, we need to handle the fact that we don't have historical context
    # We'll use the most recent historical data to compute lag features
    print("\n4️⃣ Loading recent historical data for lag features...")
    try:
        import hopsworks
        from dotenv import load_dotenv
        
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            load_dotenv()
            api_key = os.getenv("HOPSWORKS_API_KEY")
        
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store(name='aqi_predictor_qartzai_featurestore')
        fg = fs.get_feature_group("qartzai_2", version=1)
        historical_df = fg.read()
        
        # Convert and clean datetime
        historical_df["datetime"] = pd.to_datetime(historical_df["datetime"])
        if historical_df["datetime"].dt.tz is not None:
            historical_df["datetime"] = historical_df["datetime"].dt.tz_localize(None)
        
        # Get last 24 hours for lag calculations
        historical_df = historical_df.sort_values("datetime").tail(24)
        print(f"✅ Loaded {len(historical_df)} hours of recent historical data")
        
    except Exception as e:
        print(f"⚠️ Could not load historical data from Hopsworks: {e}")
        print("Using forecast data only (lag features may be less accurate)")
        historical_df = None
    
    # 5. Combine historical + forecast for proper lag calculation
    if historical_df is not None:
        # Combine but mark which rows are forecasts
        combined_df = pd.concat([historical_df, cleaned_df], ignore_index=True)
        combined_df = combined_df.sort_values("datetime").reset_index(drop=True)
        forecast_start_idx = len(historical_df)
    else:
        combined_df = cleaned_df
        forecast_start_idx = 0
    
    # 6. Feature engineering
    print("\n5️⃣ Applying feature engineering...")
    featured_df = add_features(combined_df)
    
    # Extract only forecast rows
    forecast_featured = featured_df.iloc[forecast_start_idx:].copy()
    forecast_featured = forecast_featured.reset_index(drop=True)
    
    print(f"✅ Generated features for {len(forecast_featured)} forecast hours")
    
    # 7. Prepare features for prediction
    print("\n6️⃣ Preparing features for prediction...")
    
    # Get feature names (exclude target and datetime)
    exclude_cols = ["aqi", "datetime", "datetime_str"]
    
    # Also drop leakage features (same as in training)
    leakage_features = ["aqi_rolling_24h", "aqi_lag_1h", "high_pollution_flag"]
    exclude_cols.extend(leakage_features)
    
    feature_cols = [col for col in forecast_featured.columns if col not in exclude_cols]
    
    X_forecast = forecast_featured[feature_cols]
    
    # Handle any remaining NaNs
    X_forecast = X_forecast.ffill().bfill().fillna(0)
    
    # Apply scaling if scaler exists
    if scaler is not None:
        X_forecast = scaler.transform(X_forecast)
    
    # 8. Generate predictions
    print("\n7️⃣ Generating AQI predictions...")
    predictions = model.predict(X_forecast)
    
    # 9. Create results dataframe
    results = pd.DataFrame({
        "datetime": forecast_featured["datetime"],
        "predicted_aqi": predictions
    })
    
    # Add date column for aggregation
    results["date"] = results["datetime"].dt.date
    
    print(f"\n✅ Generated {len(results)} hourly predictions")
    print(f"   Time range: {results['datetime'].min()} to {results['datetime'].max()}")
    print(f"   AQI range: {predictions.min():.1f} - {predictions.max():.1f}")
    print(f"   Mean AQI: {predictions.mean():.1f}")
    
    # 10. Save predictions
    print("\n8️⃣ Saving predictions...")
    predictions_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/predictions"))
    os.makedirs(predictions_dir, exist_ok=True)
    
    output_file = os.path.join(predictions_dir, f"next_{forecast_days}_days_predictions.csv")
    results.to_csv(output_file, index=False)
    print(f"💾 Saved predictions → {output_file}")
    
    # Optional: Upload to Hopsworks
    # try:
    #     from upload_hopsworks import upload_to_hopsworks
    #     upload_to_hopsworks(results, feature_group="qartzai_predictions", version=1)
    #     print("✅ Uploaded predictions to Hopsworks")
    # except Exception as e:
    #     print(f"⚠️ Could not upload to Hopsworks: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Prediction pipeline completed successfully!\n")
    
    return results


if __name__ == "__main__":
    predictions = generate_predictions(forecast_days=3)
    
    if predictions is not None:
        print("\n📊 Sample Predictions (First 24 hours):")
        print(predictions.head(24)[["datetime", "predicted_aqi"]].to_string(index=False))
        
        print("\n📈 Daily Average Predictions:")
        daily_avg = predictions.groupby("date")["predicted_aqi"].mean().reset_index()
        print(daily_avg.to_string(index=False))
