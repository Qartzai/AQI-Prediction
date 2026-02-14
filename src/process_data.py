import os
import pandas as pd
from config import PROCESSED_PATH, SAVE_LOCAL


def process_latest_json(raw_df):
    """
    Process the combined air quality + weather dataframe into a structured format.
    This version no longer reads JSON from disk — it uses the dataframe
    returned by fetch_api_data() in the automated pipeline.
    """

    if raw_df is None or raw_df.empty:
        raise ValueError("❌ Empty or invalid raw dataframe passed to process_latest_json()")

    # 1. Work on a copy
    df = raw_df.copy()

    # 2. Standardize datetime column
    if "time" in df.columns and "datetime" not in df.columns:
        df.rename(columns={"time": "datetime"}, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. column names to lowercase
    df.columns = df.columns.str.lower()

    # 4. Save locally if configured
    if SAVE_LOCAL:
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        out_file = os.path.join(
            PROCESSED_PATH,
            f"processed_{df['datetime'].dt.date.min()}.csv"
        )
        df.to_csv(out_file, index=False)
        print(f"✅ Processed data saved → {out_file}")
    else:
        print("⚙️ Skipping local save (cloud/CI mode).")

    print(f"📊 Processed DataFrame shape: {df.shape}")
    return df


# --- Run standalone test ---
if __name__ == "__main__":
    import json

    # Example test file 
    sample_path = os.path.join("data", "raw", "raw_combined_sample.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            sample_data = json.load(f)
        sample_df = pd.concat([
            pd.DataFrame(sample_data.get("air_quality", {}).get("hourly", {})),
            pd.DataFrame(sample_data.get("weather", {}).get("hourly", {}))
        ], axis=1)
        processed = process_latest_json(sample_df)
        print(processed.head())
    else:
        print("⚠️ No sample file found for standalone test.")