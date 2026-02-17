"""
Quick script to check current AQI for Karachi in both US EPA and European standards.
"""
import requests
from datetime import datetime
from src.aqi_utils import compute_aqi_from_row, calculate_european_aqi

# Karachi coordinates
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

# Fetch current air quality data
AIR_QUALITY_URL = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={KARACHI_LAT}&longitude={KARACHI_LON}&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&timezone=Asia/Karachi"

WEATHER_URL = f"https://api.open-meteo.com/v1/forecast?latitude={KARACHI_LAT}&longitude={KARACHI_LON}&current=temperature_2m,relative_humidity_2m&timezone=Asia/Karachi"

def get_aqi_category_us(aqi):
    """Get US EPA AQI category and color"""
    if aqi <= 50:
        return "Good", "🟢"
    elif aqi <= 100:
        return "Moderate", "🟡"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi <= 200:
        return "Unhealthy", "🔴"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣"
    else:
        return "Hazardous", "🟤"

def get_aqi_category_eu(eaqi):
    """Get European EAQI category and color"""
    if eaqi <= 25:
        return "Good", "🟢"
    elif eaqi <= 50:
        return "Fair", "💚"
    elif eaqi <= 75:
        return "Moderate", "🟡"
    elif eaqi <= 100:
        return "Poor", "🟠"
    elif eaqi <= 150:
        return "Very Poor", "🔴"
    else:
        return "Extremely Poor", "🟤"

print("\n" + "="*60)
print("🌍 CURRENT AIR QUALITY INDEX - KARACHI, PAKISTAN")
print("="*60)
print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60 + "\n")

try:
    # Fetch air quality data
    aq_response = requests.get(AIR_QUALITY_URL, timeout=10)
    aq_response.raise_for_status()
    aq_data = aq_response.json()
    
    # Fetch weather data for temperature
    wx_response = requests.get(WEATHER_URL, timeout=10)
    wx_response.raise_for_status()
    wx_data = wx_response.json()
    
    # Extract current values
    current_aq = aq_data.get("current", {})
    current_wx = wx_data.get("current", {})
    
    # Prepare data row for AQI calculation
    row = {
        "pm10": current_aq.get("pm10"),
        "pm2_5": current_aq.get("pm2_5"),
        "carbon_monoxide": current_aq.get("carbon_monoxide"),
        "nitrogen_dioxide": current_aq.get("nitrogen_dioxide"),
        "ozone": current_aq.get("ozone"),
        "sulphur_dioxide": current_aq.get("sulphur_dioxide"),
        "temperature_2m": current_wx.get("temperature_2m", 25.0),
    }
    
    # Display raw pollutant concentrations
    print("📊 POLLUTANT CONCENTRATIONS")
    print("-" * 60)
    print(f"PM2.5:              {row['pm2_5']:.1f} µg/m³" if row['pm2_5'] else "PM2.5:              N/A")
    print(f"PM10:               {row['pm10']:.1f} µg/m³" if row['pm10'] else "PM10:               N/A")
    print(f"Carbon Monoxide:    {row['carbon_monoxide']:.1f} µg/m³" if row['carbon_monoxide'] else "Carbon Monoxide:    N/A")
    print(f"Nitrogen Dioxide:   {row['nitrogen_dioxide']:.1f} µg/m³" if row['nitrogen_dioxide'] else "Nitrogen Dioxide:   N/A")
    print(f"Ozone:              {row['ozone']:.1f} µg/m³" if row['ozone'] else "Ozone:              N/A")
    print(f"Sulphur Dioxide:    {row['sulphur_dioxide']:.1f} µg/m³" if row['sulphur_dioxide'] else "Sulphur Dioxide:    N/A")
    print(f"Temperature:        {row['temperature_2m']:.1f}°C" if row['temperature_2m'] else "Temperature:        N/A")
    print()
    
    # Calculate US EPA AQI
    us_aqi_result = compute_aqi_from_row(row, temp_c=row['temperature_2m'])
    us_aqi = us_aqi_result.get("aqi")
    
    # Calculate European EAQI
    eu_result = calculate_european_aqi(row)
    eu_aqi = eu_result.get("european_aqi")
    
    # Display US EPA AQI
    print("🇺🇸 US EPA AIR QUALITY INDEX")
    print("-" * 60)
    if us_aqi is not None:
        category, icon = get_aqi_category_us(us_aqi)
        print(f"{icon} AQI Value:  {us_aqi}")
        print(f"   Category:   {category}")
        print(f"\n   Sub-indices:")
        print(f"   - PM2.5:  {us_aqi_result.get('aqi_pm25', 'N/A')}")
        print(f"   - PM10:   {us_aqi_result.get('aqi_pm10', 'N/A')}")
        print(f"   - CO:     {us_aqi_result.get('aqi_co', 'N/A')}")
        print(f"   - NO2:    {us_aqi_result.get('aqi_no2', 'N/A')}")
        print(f"   - O3:     {us_aqi_result.get('aqi_o3', 'N/A')}")
        print(f"   - SO2:    {us_aqi_result.get('aqi_so2', 'N/A')}")
    else:
        print("❌ Could not calculate US EPA AQI")
    print()
    
    # Display European EAQI
    print("🇪🇺 EUROPEAN AIR QUALITY INDEX (EAQI)")
    print("-" * 60)
    if eu_aqi is not None:
        category = eu_result.get('eaqi_category', 'Unknown')
        level = eu_result.get('eaqi_level', 0)
        icon = get_aqi_category_eu(eu_aqi)[1]
        print(f"{icon} EAQI Value: {eu_aqi:.0f}")
        print(f"   Category:   {category} (Level {level})")
        print(f"\n   Sub-indices:")
        print(f"   - PM2.5:  {eu_result.get('eaqi_pm25', 0):.0f}")
        print(f"   - PM10:   {eu_result.get('eaqi_pm10', 0):.0f}")
        print(f"   - NO2:    {eu_result.get('eaqi_no2', 0):.0f}")
        print(f"   - O3:     {eu_result.get('eaqi_o3', 0):.0f}")
        print(f"   - SO2:    {eu_result.get('eaqi_so2', 0):.0f}")
    else:
        print("❌ Could not calculate European EAQI")
    
    print("\n" + "="*60)
    print("✅ Data fetched successfully from Open-Meteo API")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"\n❌ Error fetching or calculating AQI: {e}\n")
