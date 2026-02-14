# Purpose: Add advanced engineered features + feature refinement

import pandas as pd
import numpy as np
import os

# Safe import for compute_aqi function
try:
    from src.aqi_utils import compute_aqi_from_row
    from src.config import SAVE_LOCAL
except Exception:
    from aqi_utils import compute_aqi_from_row
    from config import SAVE_LOCAL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AQI, time-based, and derived features for ML training.
    Includes both Phase-1 (feature creation) and Phase-2 (feature refinement from EDA-2).
    """

    df = df.copy()

    #1. Normalize datetime column
    if "time" in df.columns and "datetime" not in df.columns:
        df.rename(columns={"time": "datetime"}, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2.Compute AQI (using compute_aqi_from_row)
    print("⚙️ Computing AQI and sub-indices...")
    aqi_results = df.apply(lambda row: compute_aqi_from_row(row), axis=1)

    # Expand dict results into DataFrame
    if isinstance(aqi_results.iloc[0], dict):
        aqi_expanded = pd.DataFrame(list(aqi_results))
        aqi_expanded = aqi_expanded.apply(pd.to_numeric, errors="coerce")
        df = pd.concat([df.reset_index(drop=True), aqi_expanded.reset_index(drop=True)], axis=1)
    else:
        df["aqi"] = pd.to_numeric(aqi_results, errors="coerce")

    # Ensure 'aqi' column exists
    if "aqi" not in df.columns:
        sub_cols = [c for c in df.columns if c.startswith("aqi_")]
        df["aqi"] = df[sub_cols].max(axis=1) if sub_cols else np.nan

    # Drop rows where AQI couldn't be computed
    df.dropna(subset=["aqi"], inplace=True)

    # 3. Time-based features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    # Cyclic encoding (sin/cos for hour)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 4. Derived features 
    df["aqi_change_rate"] = df["aqi"].diff()
    df["aqi_roll_mean_3h"] = df["aqi"].rolling(window=3, min_periods=1).mean()
    df["aqi_roll_mean_6h"] = df["aqi"].rolling(window=6, min_periods=1).mean()
    df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()

    # 5. Lag features
    for lag in [1, 3, 6]:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)

    # 6. Pollutant ratio features
    df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)

    # 7. Meteorological combination features
    df["temp_humidity_ratio"] = df["temperature_2m"] / (df["relative_humidity_2m"] + 1e-6)
    df["wind_effect"] = df["wind_speed_10m"] * np.cos(np.deg2rad(df["wind_direction_10m"]))

    # 8. High pollution flag
    df["high_pollution_flag"] = np.where(df["aqi"] > 150, 1, 0)

    # 9. Handle NaNs from lags/ratios 
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print("🧠 Base feature engineering complete! Proceeding with EDA-2 refinement...")

    # =============================================================
    # ✨ PHASE-2: Feature Refinement (EDA-2 Logic)
    # =============================================================

    drop_cols = [
        # Redundant pollutant sub-indices
        'aqi_pm25', 'aqi_pm10', 'no2_ppb', 'o3_ppb', 'so2_ppb', 'co_ppm',
        'aqi_no2', 'aqi_so2', 'aqi_co', 'aqi_o3',

        # Weak correlation / low variance
        'aqi_o3_1h', 'hour_cos', 'wind_direction_10m',

        # Redundant time-based aggregates
        'aqi_roll_mean_3h', 'aqi_roll_mean_6h', 'aqi_lag_3h', 'aqi_lag_6h'
    ]

    df_refined = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    print("✅ Feature refinement done.")
    print(f"Final selected shape: {df_refined.shape}")
    print(f"Final columns: {df_refined.columns.tolist()}")

    return df_refined


# --- Run standalone test safely ---
# if __name__ == "__main__":
#     PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "final", "clean_merged_karachi.csv")
#     OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "final", "final_selected_features.csv")

#     if os.path.exists(INPUT_PATH):
#         df = pd.read_csv(INPUT_PATH)
#         print(f"📂 Found input file: {INPUT_PATH}")

#         print("⚙️ Running full feature engineering (Phase 1 + Phase 2)...")
#         final_df = add_features(df)

#         if SAVE_LOCAL:
#             os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#             final_df.to_csv(OUTPUT_PATH, index=False)
#             print(f"💾 Saved refined dataset → {OUTPUT_PATH}")
#         else:
#             print("⚙️ Skipping local save (cloud mode).")

#     else:
#         print(f"❌ Input file not found → {INPUT_PATH}")