import os
import requests
import json
from datetime import datetime
from config import AIR_QUALITY_URL, WEATHER_FORECAST_URL, RAW_PATH, SAVE_LOCAL

def fetch_api_data(url: str):
    """Fetch data from given API endpoint."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def save_combined_raw(aq_data: dict, wx_data: dict, folder_path: str):
    """Save Air Quality + Weather data together in one JSON file."""
    if SAVE_LOCAL:
        os.makedirs(folder_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"raw_combined_{timestamp}.json"
        filepath = os.path.join(folder_path, filename)

        combined = {
            "metadata": {
                "saved_at": timestamp,
                "source": "Open-Meteo APIs",
            },
            "air_quality": aq_data,
            "weather": wx_data
        }

        with open(filepath, "w") as f:
            json.dump(combined, f, indent=2)

        print(f"Saved combined data → {filepath}")
    else:
        print("Skipping local save (running in cloud/CI mode).")

def main():
    # Fetch Air Quality
    aq_data = fetch_api_data(AIR_QUALITY_URL)

    # Fetch Weather
    wx_data = fetch_api_data(WEATHER_FORECAST_URL)

    if aq_data and wx_data:
        save_combined_raw(aq_data, wx_data, RAW_PATH)
    else:
        print("Could not fetch one or more APIs, skipping save.")

if __name__ == "__main__":
    main()