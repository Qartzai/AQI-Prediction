"""
Verify what the dashboard should display - check current AQI and predictions
"""
import pandas as pd
import requests
import sys
import os

sys.path.insert(0, 'src')
from config import LAT, LON
from aqi_utils import compute_aqi_from_row, calculate_european_aqi

print("="*70)
print("DASHBOARD DATA VERIFICATION")
print("="*70)

# 1. Check Current AQI from API
print("\n📡 CURRENT AIR QUALITY FROM API")
print("-" * 70)

aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&timezone=Asia/Karachi"
wx_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=Asia/Karachi"

aq_data = requests.get(aq_url).json()["current"]
wx_data = requests.get(wx_url).json()["current"]

current_data = {**aq_data, **wx_data}

print(f"PM2.5:           {current_data['pm2_5']:.1f} µg/m³")
print(f"PM10:            {current_data['pm10']:.1f} µg/m³")
print(f"CO:              {current_data['carbon_monoxide']:.1f} µg/m³")
print(f"NO2:             {current_data['nitrogen_dioxide']:.1f} µg/m³")
print(f"O3:              {current_data['ozone']:.1f} µg/m³")
print(f"SO2:             {current_data['sulphur_dioxide']:.1f} µg/m³")
print(f"Temperature:     {current_data['temperature_2m']:.1f}°C")
print(f"Humidity:        {current_data['relative_humidity_2m']:.0f}%")
print(f"Wind Speed:      {current_data['wind_speed_10m']:.1f} km/h")

# Calculate AQI
us_result = compute_aqi_from_row(current_data, temp_c=current_data['temperature_2m'])
eu_result = calculate_european_aqi(current_data)

print(f"\n🇺🇸 US EPA AQI:    {us_result['aqi']}")
print(f"🇪🇺 European EAQI: {eu_result['european_aqi']:.0f}")

# Convert gases to ppb/ppm for display
MW_NO2 = 46.0055
MW_O3 = 48.00
MW_SO2 = 64.066
MW_CO = 28.01

def ugm3_to_ppb(ugm3, mw, temp_c=25.0):
    T_K = temp_c + 273.15
    ppb = ugm3 * (24.45 / mw) * (T_K / 298.15)
    return ppb

temp_c = current_data['temperature_2m']
print(f"\n📊 POLLUTANTS IN DISPLAY UNITS:")
print(f"NO2:  {ugm3_to_ppb(current_data['nitrogen_dioxide'], MW_NO2, temp_c):.1f} ppb")
print(f"O3:   {ugm3_to_ppb(current_data['ozone'], MW_O3, temp_c):.1f} ppb")
print(f"SO2:  {ugm3_to_ppb(current_data['sulphur_dioxide'], MW_SO2, temp_c):.1f} ppb")
print(f"CO:   {ugm3_to_ppb(current_data['carbon_monoxide'], MW_CO, temp_c)/1000:.1f} ppm")

# 2. Check Predictions
print("\n" + "="*70)
print("📅 PREDICTIONS DATA")
print("-" * 70)

predictions_file = "data/predictions/next_3_days_predictions.csv"
if os.path.exists(predictions_file):
    df_pred = pd.read_csv(predictions_file)
    df_pred["datetime"] = pd.to_datetime(df_pred["datetime"])
    df_pred["date"] = pd.to_datetime(df_pred["date"])
    
    print(f"Total predictions: {len(df_pred)} hours")
    print(f"Date range: {df_pred['datetime'].min()} to {df_pred['datetime'].max()}")
    print(f"\nAQI range: {df_pred['predicted_aqi'].min():.1f} to {df_pred['predicted_aqi'].max():.1f}")
    
    print("\n📊 DAILY AVERAGES:")
    daily = df_pred.groupby('date')['predicted_aqi'].agg(['mean', 'min', 'max'])
    for date, row in daily.iterrows():
        print(f"  {date.strftime('%Y-%m-%d (%A)')}: avg={row['mean']:.0f}, min={row['min']:.0f}, max={row['max']:.0f}")
    
    print(f"\n🔍 FIRST 5 HOURS:")
    print(df_pred[['datetime', 'predicted_aqi']].head().to_string(index=False))
else:
    print("❌ Predictions file not found!")

print("\n" + "="*70)
print("✅ VERIFICATION COMPLETE")
print("="*70)
print("\n💡 To refresh Streamlit dashboard:")
print("   1. Stop the running dashboard (Ctrl+C)")
print(f"   2. Run: streamlit run streamlit_app/app.py")
print(f"   3. Or press 'R' or 'Ctrl+R' in browser to reload")
print("="*70 + "\n")
