import pandas as pd
import numpy as np
import os

try:
    from src.config import SAVE_LOCAL
except Exception:
    from config import SAVE_LOCAL


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans merged AQI + weather dataset.
    Based on EDA-1 + minimal future-safe improvements:
      • Convert datetime
      • Handle missing values safely
      • Cap pollutant outliers
    """
    df = df.copy()

    #1. Datetime normalization 
    if "time" in df.columns and "datetime" not in df.columns:
        df.rename(columns={"time": "datetime"}, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Convert numeric columns 
    pollutant_cols = ["pm10", "pm2_5", "carbon_monoxide",
                      "nitrogen_dioxide", "ozone", "sulphur_dioxide"]
    weather_cols = ["temperature_2m", "relative_humidity_2m",
                    "wind_speed_10m", "wind_direction_10m"]

    for col in pollutant_cols + weather_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Safe fill for small missing gaps 
    df = df.ffill().bfill()

    # 4. Outlier capping (EDA-1 logic)
    outlier_cols = ["pm2_5", "pm10", "carbon_monoxide"]
    for col in outlier_cols:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df[col] = np.clip(df[col], lower, upper)

    # 5. Drop columns if >50% NaN (safety net)
    threshold = len(df) * 0.5
    df.dropna(axis=1, thresh=threshold, inplace=True)

    print("Data cleaning complete (EDA-1 + future-safe handling).")
    print(f"Final cleaned shape: {df.shape}")

    return df


# --- Run standalone test ---
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "final", "merged_karachi.csv")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "final", "clean_merged_karachi.csv")

    if os.path.exists(INPUT_PATH):
        df = pd.read_csv(INPUT_PATH)
        cleaned_df = clean_data(df)

        if SAVE_LOCAL:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            cleaned_df.to_csv(OUTPUT_PATH, index=False)
            print(f"💾 Cleaned data saved → {OUTPUT_PATH}")
        else:
            print("⚙️ Skipping local save (cloud mode).")
    else:
        print(f"❌ File not found → {INPUT_PATH}")