"""
Check the last 24 hours of AQI data from Hopsworks to find the 500 spike
"""
import pandas as pd
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('HOPSWORKS_API_KEY')

print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store(name='aqi_predictor_qartzai_featurestore')
fg = fs.get_feature_group('qartzai_2', version=1)
df = fg.read()

print(f"\nColumns in data: {df.columns.tolist()}")
print(f"Total rows: {len(df)}")

# Find datetime column
datetime_col = None
for col in df.columns:
    if 'datetime' in col.lower() or 'time' in col.lower():
        datetime_col = col
        break

if datetime_col:
    print(f"Using datetime column: {datetime_col}")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)
    
    # Get last 24 hours
    last_24 = df.tail(24).copy()
    
    print(f"\n📊 LAST 24 HOURS AQI VALUES:")
    print("="*70)
    print(last_24[[datetime_col, 'aqi']].to_string(index=False))
    
    print(f"\n📈 STATISTICS:")
    print(f"   Min:     {last_24['aqi'].min():.0f}")
    print(f"   Max:     {last_24['aqi'].max():.0f}")
    print(f"   Mean:    {last_24['aqi'].mean():.0f}")
    print(f"   Median:  {last_24['aqi'].median():.0f}")
    
    # Find outliers (AQI > 200)
    outliers = last_24[last_24['aqi'] > 200]
    if len(outliers) > 0:
        print(f"\n⚠️  OUTLIERS (AQI > 200):")
        print(outliers[[datetime_col, 'aqi', 'pm2_5', 'pm10']].to_string(index=False))
    
    # Check for 500 values
    extreme = last_24[last_24['aqi'] >= 500]
    if len(extreme) > 0:
        print(f"\n🚨 EXTREME VALUES (AQI = 500):")
        print(extreme[[datetime_col, 'aqi', 'pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide']].to_string(index=False))
else:
    print("❌ Could not find datetime column!")
    print(f"Columns: {df.columns.tolist()}")
