LAT = 24.8607   # Karachi latitude
LON = 67.0011   # Karachi longitude

# Base URLs for latest fetch
AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LAT}&longitude={LON}"
    f"&forecast_days=1"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
)

WEATHER_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    "&forecast_days=1"
)

# base urls for historical data
from datetime import datetime, timedelta
start_date = (datetime.now() - timedelta(days=360)).strftime("%Y-%m-%d")
end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
aq_historic_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
    )
weather_historic_url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )

# Data path
RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed"
HIST_PATH = "data/historical"

import os
SAVE_LOCAL = os.getenv("SAVE_LOCAL", "false").lower() in ("1", "true", "yes")