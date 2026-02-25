"""
Generate local feature data without Hopsworks
Processes historical data and creates final_selected_features.csv
"""
import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from process_features import process_features

print("=" * 70)
print("🔄 Generating Local Feature Data")
print("=" * 70)

# Load historical data
historical_path = "data/historical/historical_karachi_1y.csv"
if not os.path.exists(historical_path):
    print(f"❌ Historical data not found: {historical_path}")
    print("   Run: python src/backfill_historical.py")
    exit(1)

print(f"\n1️⃣ Loading historical data...")
df = pd.read_csv(historical_path)
print(f"   ✅ Loaded {len(df)} rows")

# Process features
print(f"\n2️⃣ Processing features...")
df_featured = process_features(df)
print(f"   ✅ Created {df_featured.shape[1]} features")

# Drop leakage features
leakage_features = ["aqi_rolling_24h", "aqi_lag_1h", "high_pollution_flag"]
df_featured.drop(columns=[col for col in leakage_features if col in df_featured.columns], 
                 inplace=True, errors="ignore")
print(f"   ✅ Dropped leakage features")

# Save to final directory
output_dir = "data/final"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "final_selected_features.csv")

df_featured.to_csv(output_path, index=False)
print(f"\n3️⃣ Saved features → {output_path}")
print(f"   📏 Shape: {df_featured.shape}")
print(f"   📅 Columns: {list(df_featured.columns)[:10]}...")

if "datetime" in df_featured.columns:
    df_featured["datetime"] = pd.to_datetime(df_featured["datetime"])
    print(f"   ⏰ Time range: {df_featured['datetime'].min()} to {df_featured['datetime'].max()}")

if "aqi" in df_featured.columns:
    print(f"   🌫️  AQI range: {df_featured['aqi'].min():.1f} - {df_featured['aqi'].max():.1f}")

print("\n" + "=" * 70)
print("✅ Local data ready! You can now run:")
print("   streamlit run streamlit_app/app.py")
print("=" * 70)
