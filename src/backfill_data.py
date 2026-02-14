import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from config import (
    aq_historic_url,
    weather_historic_url,
    HIST_PATH
)
from process_features import add_features
from clean_data import clean_data
from process_data import process_latest_json


def fetch_archive():
    """Fetch historical air quality + weather data from Open-Meteo Archive API"""
    print("Fetching historical air quality and weather data...")

    # Request APIs (360 days = large dataset, need longer timeout)
    aq_response = requests.get(aq_historic_url, timeout=120)
    wx_response = requests.get(weather_historic_url, timeout=120)

    # Validate API responses
    aq_response.raise_for_status()
    wx_response.raise_for_status()

    aq_data = aq_response.json().get("hourly", {})
    wx_data = wx_response.json().get("hourly", {})

    # Convert to DataFrame
    df_aq = pd.DataFrame(aq_data)
    df_wx = pd.DataFrame(wx_data)

    # Merge on datetime (common column = 'time') 
    df = pd.merge(df_aq, df_wx, on="time", how="inner")

    # Rename for consistency 
    df.rename(columns={"time": "datetime"}, inplace=True)

    print(f"✅ Retrieved {len(df)} hourly records of historical data.")
    return df


def backfill(years=1):
    """Fetch and process historical data for given number of years."""
    print(f"\nRunning backfill for ~{years} year(s)...")

    # --- Fetch combined historical data ---
    raw_df = fetch_archive()

    # --- Process: Clean → Feature Engineering ---
    print("\n⚙️ Processing raw historical data...")
    processed_df = process_latest_json(raw_df)
    
    print("\n🧹 Cleaning processed data...")
    cleaned_df = clean_data(processed_df)
    
    print("\n🧠 Applying feature engineering...")
    featured_df = add_features(cleaned_df)
    print(f"✅ Feature engineering complete! Shape: {featured_df.shape}")

    # --- Save to historical folder ---
    os.makedirs(HIST_PATH, exist_ok=True)
    out_file = os.path.join(HIST_PATH, f"historical_karachi_{years}y.csv")
    featured_df.to_csv(out_file, index=False)

    print(f"Saved historical dataset → {out_file}")
    print(f"Total rows: {len(featured_df)} | Columns: {list(featured_df.columns)}")
    
    # --- Upload to Hopsworks qartzai_2 ---
    print("\n📦 Uploading backfill data to Hopsworks (qartzai_2)...")
    from upload_hopsworks import upload_to_hopsworks
    upload_to_hopsworks(featured_df, feature_group="qartzai_2", version=1)


if __name__ == "__main__":
    backfill(years=1)