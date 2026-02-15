import hopsworks
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime

try:
    from src.config import SAVE_LOCAL
except Exception:
    from config import SAVE_LOCAL


def upload_to_hopsworks(df: pd.DataFrame = None, feature_group: str = "aqi_features", version: int = 2):
    """
    Upload final processed feature DataFrame to Hopsworks Feature Store.
    If df is not provided, it loads the latest 'final_selected_features.csv'.
    
    Args:
        df: DataFrame to upload
        feature_group: Name of the feature group (default: "aqi_features")
        version: Version of the feature group (default: 2)
    """

    print("🔗 Connecting to Hopsworks Feature Store...")

    # 1. Load environment variables (API key)
    # Try environment variable first (for CI/CD), then .env file (for local)
    api_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not api_key:
        # Try loading from .env file for local development
        load_dotenv()
        api_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not api_key:
        raise ValueError(
            "❌ Missing HOPSWORKS_API_KEY!\n"
            "   For local: Add HOPSWORKS_API_KEY to .env file\n"
            "   For GitHub Actions: Add HOPSWORKS_API_KEY to repository secrets\n"
            "   Visit: Settings → Secrets and variables → Actions → New repository secret"
        )

    # 2. Authenticate & connect to project
    try:
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store(name='aqi_predictor_qartzai_featurestore')
        print("✅ Connected to Hopsworks Feature Store")
    except Exception as e:
        print(f"\n❌ Failed to connect to Hopsworks!")
        print(f"   Error: {str(e)}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Generate a new API key at https://app.hopsworks.ai/")
        print(f"   2. For GitHub: Update secret at Settings → Secrets → Actions")
        print(f"   3. For local: Update .env file with new key")
        raise

    # 3. Load DataFrame (if not passed) 
    if df is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(BASE_DIR, "data", "final", "final_selected_features.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        df = pd.read_csv(file_path)
        print(f"📂 Loaded dataset → {file_path}")

    print(f"📊 Dataset shape before upload: {df.shape}")

    # 4. Datetime handling 
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["datetime_str"] = df["datetime"].astype(str)
        df.drop(columns=["datetime"], inplace=True)
    elif "datetime_str" not in df.columns:
        raise ValueError("❌ Missing datetime column in DataFrame")

    # 5. Drop extra columns not in FG schema
    drop_extras = ["year", "month_num", "day_num"]
    df = df.drop(columns=[c for c in drop_extras if c in df.columns], errors="ignore")

    # 6. Enforce correct dtypes (align with FG schema)
    int_cols = ["month", "hour", "day", "weekday", "high_pollution_flag"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.int64)

    # 7. Use provided Feature Group metadata
    # feature_group and version passed as parameters

    # 8. Get or create Feature Group
    fg = fs.get_or_create_feature_group(
        name=feature_group,
        version=version,
        primary_key=["datetime_str"],
        description="Karachi AQI selected features (daily ingestion)",
        online_enabled=True
    )

    # 9. Insert into Feature Store 
    print("🚀 Uploading to Hopsworks Feature Store...")
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Successfully uploaded {len(df)} rows to Feature Group → '{feature_group}_v{version}'")

    # 10. local snapshot
    if SAVE_LOCAL:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = os.path.join(BASE_DIR, "data", "final", "uploaded_snapshot.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"💾 Snapshot saved locally → {out_path}")
    else:
        print("⚙️ Skipping local snapshot save (cloud mode).")

    print("🎉 Upload complete.")
    return df


# --- Run standalone test safely ---
if __name__ == "__main__":
    upload_to_hopsworks()