import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from config import (
    aq_historic_url,
    weather_historic_url,
    HIST_PATH
)
from process_data import add_features


def fetch_archive():
    """Fetch historical air quality + weather data from Open-Meteo Archive API"""
    print("Fetching historical air quality and weather data...")

    # Request APIs 
    aq_response = requests.get(aq_historic_url, timeout=15)
    wx_response = requests.get(weather_historic_url, timeout=15)

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
    df = fetch_archive()

    # --- Save to historical folder ---
    os.makedirs(HIST_PATH, exist_ok=True)
    out_file = os.path.join(HIST_PATH, f"historical_karachi_{years}y.csv")
    df.to_csv(out_file, index=False)

    print(f"Saved historical dataset → {out_file}")
    print(f"Total rows: {len(df)} | Columns: {list(df.columns)}")


if __name__ == "__main__":
    backfill(years=1)