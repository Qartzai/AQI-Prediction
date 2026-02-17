import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import os
from datetime import timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# PAGE CONFIG 
st.set_page_config(
    page_title="Karachi AQI Prediction Bot",
    page_icon="🌤",
    layout="wide"
)

# INLINE STYLING 
st.markdown("""
    <style>
        body {
            background-color: #f6fafc;
            color: #222;
            font-family: "Inter", sans-serif;
        }
        .main-title {
            text-align: center;
            color: #1565C0;
            font-weight: 800;
            font-size: 32px;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #388E3C;
            font-size: 16px;
            margin-bottom: 25px;
        }
        .metric-card {
            padding: 15px;
            background-color: #E3F2FD;
            border-left: 5px solid #1565C0;
            border-radius: 10px;
            box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
        }
        .chart-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
        }
        .footer {
            text-align: center;
            color: gray;
            font-size: 13px;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# TITLE SECTION 
st.markdown("<h1 class='main-title'>Karachi AQI Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time and 3-day Air Quality predictions powered by ML & Hopsworks Feature Store.</p>", unsafe_allow_html=True)

# CONNECT TO HOPSWORKS
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

try:
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store(name='aqi_predictor_qartzai_featurestore')
    fg = fs.get_feature_group("qartzai_2", version=1)
    df = fg.read()
    
    # Make datetime timezone-naive to avoid comparison errors
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
    
    st.success("Connected to Hopsworks and fetched latest data.")
except Exception as e:
    st.error(f"⚠ Could not fetch data from Hopsworks: {e}")
    st.info("Using local fallback data...")
    df = pd.read_csv("../data/final/final_selected_features.csv")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

# DATA PREPARATION
if "datetime_str" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime_str"])
    df.drop(columns=["datetime_str"], inplace=True)

df = df.sort_values("datetime").reset_index(drop=True)

# Drop leakage features
leakage_features = ["aqi_rolling_24h", "aqi_lag_1h", "high_pollution_flag"]
df.drop(columns=[col for col in leakage_features if col in df.columns], inplace=True, errors="ignore")

# LOAD PREDICTIONS from Hopsworks
# LOAD PREDICTIONS - prefer local file for development
has_predictions = False
predictions_df = None

# First try local file (most up-to-date during development)
predictions_path = os.path.join(os.path.dirname(__file__), "../data/predictions/next_3_days_predictions.csv")
try:
    if os.path.exists(predictions_path):
        predictions_df = pd.read_csv(predictions_path)
        predictions_df["datetime"] = pd.to_datetime(predictions_df["datetime"])
        st.success("Loaded predictions from local file.")
        has_predictions = True
except Exception as local_error:
    st.warning(f"Could not load local predictions: {local_error}")

# Fallback to Hopsworks if local file not available
if not has_predictions:
    try:
        fg_predictions = fs.get_feature_group("qartzai_predictions", version=1)
        predictions_df = fg_predictions.read()
        predictions_df["datetime"] = pd.to_datetime(predictions_df["datetime"])
        if predictions_df["datetime"].dt.tz is not None:
            predictions_df["datetime"] = predictions_df["datetime"].dt.tz_localize(None)
        st.success("Loaded future AQI predictions from Hopsworks.")
        has_predictions = True
    except Exception as e:
        st.warning(f"No predictions found. Wait for daily prediction pipeline to run.")
        has_predictions = False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_aqi_category(aqi_value):
    """Get US EPA AQI category and color"""
    if aqi_value <= 50:
        return "Good", "#66BB6A"
    elif aqi_value <= 100:
        return "Moderate", "#FDD835"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "#FB8C00"
    elif aqi_value <= 200:
        return "Unhealthy", "#E53935"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#8E24AA"
    else:
        return "Hazardous", "#6D4C41"

def get_eaqi_category(eaqi_value):
    """Get European AQI category and color"""
    if eaqi_value <= 25:
        return "Good", "#66BB6A", 1
    elif eaqi_value <= 50:
        return "Fair", "#9CCC65", 2
    elif eaqi_value <= 75:
        return "Moderate", "#FDD835", 3
    elif eaqi_value <= 100:
        return "Poor", "#FB8C00", 4
    elif eaqi_value <= 150:
        return "Very Poor", "#E53935", 5
    else:
        return "Extremely Poor", "#6D4C41", 6

def get_health_recommendation(us_aqi, eu_aqi):
    """Get health recommendations based on AQI"""
    if us_aqi <= 50:
        return "Air quality is satisfactory. Ideal for outdoor activities."
    elif us_aqi <= 100:
        return "Acceptable air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion."
    elif us_aqi <= 150:
        return "Sensitive groups (children, elderly, respiratory issues) should limit outdoor activities."
    elif us_aqi <= 200:
        return "Everyone may experience health effects. Sensitive groups should avoid outdoor activities."
    elif us_aqi <= 300:
        return "Health alert! Everyone should avoid all outdoor physical activities."
    else:
        return "Emergency conditions! Stay indoors with air purifiers. Avoid all outdoor exposure."

def recalculate_aqi_from_pollutants(row):
    """Recalculate US EPA AQI from raw pollutant values"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
        from aqi_utils import compute_aqi_from_row
        result = compute_aqi_from_row(row, temp_c=row.get('temperature_2m', 25.0))
        return result.get('aqi', row.get('aqi', 0))
    except Exception as e:
        st.warning(f"Using stored AQI value (recalculation failed: {e})")
        return row.get('aqi', 0)

def calculate_european_aqi_simple(row):
    """Calculate European AQI from pollutants"""
    try:
        # Import the function from aqi_utils
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
        from aqi_utils import calculate_european_aqi
        return calculate_european_aqi(row)
    except Exception as e:
        st.warning(f"European AQI calculation failed: {e}")
        # Simplified fallback calculation
        pm25 = row.get("pm2_5", 0) or 0
        pm10 = row.get("pm10", 0) or 0
        
        # Simple EAQI approximation based on PM2.5
        if pm25 <= 10:
            eaqi = pm25 * 2.5
        elif pm25 <= 20:
            eaqi = 25 + (pm25 - 10) * 2.5
        elif pm25 <= 25:
            eaqi = 50 + (pm25 - 20) * 5
        elif pm25 <= 50:
            eaqi = 75 + (pm25 - 25)
        elif pm25 <= 75:
            eaqi = 100 + (pm25 - 50) * 2
        else:
            eaqi = 150 + (pm25 - 75) * 2
        
        return {"european_aqi": eaqi, "eaqi_category": "Calculated"}


if "aqi" in df.columns and len(df) > 0:
    latest = df.iloc[-1]
    
    # Recalculate AQI from raw pollutant values to ensure accuracy
    latest_aqi = recalculate_aqi_from_pollutants(latest)
    latest_datetime = latest["datetime"]
    
    # Calculate European AQI
    eaqi_result = calculate_european_aqi_simple(latest)
    european_aqi = eaqi_result.get("european_aqi", 0)
    
    st.markdown("---")
    st.subheader("Current Air Quality Status")
    st.caption(f"Last updated: {latest_datetime.strftime('%Y-%m-%d %H:%M')}")
    
    # ========= TWO-COLUMN AQI COMPARISON =========
    col_us, col_eu = st.columns(2)
    
    with col_us:
        st.markdown("### 🇺🇸 US EPA AQI")
        us_category, us_color = get_aqi_category(latest_aqi)
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, {us_color}22 0%, {us_color}44 100%); 
                        border-left: 5px solid {us_color}; padding: 20px; border-radius: 10px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h1 style='color: {us_color}; margin: 0; font-size: 3em;'>{latest_aqi:.0f}</h1>
                <p style='margin: 5px 0; font-size: 1.2em;'><b>{us_category}</b></p>
                <p style='margin: 0; font-size: 0.9em; opacity: 0.8;'>Scale: 0-500</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_eu:
        st.markdown("### 🇪🇺 European EAQI")
        eu_category, eu_color, eu_level = get_eaqi_category(european_aqi)
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, {eu_color}22 0%, {eu_color}44 100%); 
                        border-left: 5px solid {eu_color}; padding: 20px; border-radius: 10px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h1 style='color: {eu_color}; margin: 0; font-size: 3em;'>{european_aqi:.0f}</h1>
                <p style='margin: 5px 0; font-size: 1.2em;'><b>{eu_category} (Level {eu_level})</b></p>
                <p style='margin: 0; font-size: 0.9em; opacity: 0.8;'>Scale: 0-300</p>
            </div>
        """, unsafe_allow_html=True)
    
    # ========= HEALTH RECOMMENDATIONS =========
    st.markdown("### Health Recommendations")
    health_rec = get_health_recommendation(latest_aqi, european_aqi)
    st.info(health_rec)
    
    st.markdown("---")
    
    # ========= POLLUTANT BREAKDOWN =========
    st.subheader("Individual Pollutant Levels")
    
    pollutant_cols = st.columns(3)
    
    # Convert gas pollutants from µg/m³ to ppb/ppm for display
    temp_c = latest.get("temperature_2m", 25.0)
    
    # Helper function to convert µg/m³ to ppb
    def ugm3_to_ppb(ugm3, mw, temp_c=25.0):
        if ugm3 is None or ugm3 == 0:
            return 0
        T_K = temp_c + 273.15
        ppb = ugm3 * (24.45 / mw) * (T_K / 298.15) * (1013.25 / 1013.25)
        return ppb
    
    # Molecular weights
    MW_NO2 = 46.0055
    MW_O3 = 48.00
    MW_SO2 = 64.066
    MW_CO = 28.01
    
    # Convert gases
    no2_ppb = ugm3_to_ppb(latest.get("nitrogen_dioxide", 0), MW_NO2, temp_c)
    o3_ppb = ugm3_to_ppb(latest.get("ozone", 0), MW_O3, temp_c)
    so2_ppb = ugm3_to_ppb(latest.get("sulphur_dioxide", 0), MW_SO2, temp_c)
    co_ppm = ugm3_to_ppb(latest.get("carbon_monoxide", 0), MW_CO, temp_c) / 1000.0  # Convert ppb to ppm
    
    pollutants_data = [
        ("PM2.5", latest.get("pm2_5", 0), "μg/m³", 0, 100, 35.4),
        ("PM10", latest.get("pm10", 0), "μg/m³", 0, 200, 154),
        ("O₃", o3_ppb, "ppb", 0, 150, 70),
        ("NO₂", no2_ppb, "ppb", 0, 100, 53),
        ("SO₂", so2_ppb, "ppb", 0, 75, 35),
        ("CO", co_ppm, "ppm", 0, 15, 9),
    ]
    
    for idx, (name, value, unit, min_val, max_val, threshold) in enumerate(pollutants_data):
        with pollutant_cols[idx % 3]:
            percentage = min((value / threshold) * 100, 150)
            if percentage <= 50:
                bar_color = "#66BB6A"
            elif percentage <= 100:
                bar_color = "#FDD835"
            else:
                bar_color = "#E53935"
            
            st.markdown(f"""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
                    <p style='margin: 0; font-size: 0.9em; color: #666;'><b>{name}</b></p>
                    <p style='margin: 5px 0; font-size: 1.8em; font-weight: bold;'>{value:.1f} <span style='font-size: 0.5em; color: #999;'>{unit}</span></p>
                    <div style='background: #e0e0e0; height: 8px; border-radius: 4px; overflow: hidden;'>
                        <div style='background: {bar_color}; width: {min(percentage, 100)}%; height: 100%;'></div>
                    </div>
                    <p style='margin: 5px 0 0 0; font-size: 0.75em; color: #999;'>Threshold: {threshold} {unit}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========= WEATHER CONDITIONS =========
    st.subheader("Current Weather Conditions")
    
    weather_cols = st.columns(4)
    with weather_cols[0]:
        temp = latest.get("temperature_2m", 0)
        st.metric("Temperature", f"{temp:.1f}°C", help="2m above ground")
    with weather_cols[1]:
        humidity = latest.get("relative_humidity_2m", 0)
        st.metric("Humidity", f"{humidity:.0f}%", help="Relative humidity")
    with weather_cols[2]:
        wind = latest.get("wind_speed_10m", 0)
        st.metric("Wind Speed", f"{wind:.1f} km/h", help="10m above ground")
    with weather_cols[3]:
        # Calculate air quality trend from last 3 hours
        if len(df) >= 4:
            try:
                row_3h_ago = df.iloc[-4]
                aqi_3h_ago = recalculate_aqi_from_pollutants(row_3h_ago)
                aqi_change = latest_aqi - aqi_3h_ago
                trend = "↗Worsening" if aqi_change > 5 else "↘Improving" if aqi_change < -5 else "→ Stable"
            except:
                trend = "→ Stable"
        else:
            trend = "→ Stable"
        st.metric("3h Trend", trend)
    
    st.markdown("---")
    
    # ========= 24-HOUR HISTORICAL TREND =========
    st.subheader("24-Hour AQI History")
    
    if len(df) >= 24:
        last_24h = df.tail(24).copy()
        
        # Recalculate AQI for all rows to fix any incorrect stored values
        recalculated_aqi = []
        for idx, row in last_24h.iterrows():
            try:
                aqi_value = recalculate_aqi_from_pollutants(row)
                recalculated_aqi.append(aqi_value)
            except:
                # If recalculation fails, use stored value but cap at 300
                recalculated_aqi.append(min(row['aqi'], 300))
        
        last_24h['aqi_corrected'] = recalculated_aqi
        
        chart_cols = st.columns([2, 1])
        
        with chart_cols[0]:
            # Line chart with corrected AQI values
            chart_data = pd.DataFrame({
                'US EPA AQI': last_24h['aqi_corrected'].values,
                'Time': last_24h['datetime'].values
            }).set_index('Time')
            st.line_chart(chart_data, height=300)
            st.caption("AQI values recalculated from raw pollutant data for accuracy")
        
        with chart_cols[1]:
            # Statistics box
            st.markdown("""
                <div style='background: #f8f9fa; padding: 20px; border-radius: 10px;'>
                    <h4 style='margin-top: 0;'>24h Statistics</h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.metric("Minimum", f"{last_24h['aqi_corrected'].min():.0f}")
            st.metric("Maximum", f"{last_24h['aqi_corrected'].max():.0f}")
            st.metric("Average", f"{last_24h['aqi_corrected'].mean():.0f}")
    else:
        st.info("Insufficient historical data for 24-hour trend.")
    
else:
    st.warning("No current AQI data available")

# =============================================================================
# FUTURE PREDICTIONS
# =============================================================================
st.markdown("---")
st.subheader("3-Day AQI Forecast")

if has_predictions and predictions_df is not None:
    # Use real predictions from generate_predictions.py
    future_results = predictions_df.copy()
    future_results["date"] = pd.to_datetime(future_results["datetime"]).dt.date
    
    # Calculate daily statistics
    daily_stats = future_results.groupby("date").agg({
        "predicted_aqi": ["mean", "min", "max"]
    }).reset_index()
    daily_stats.columns = ["date", "avg_aqi", "min_aqi", "max_aqi"]
    
    # ========= PREDICTION SUMMARY CARDS =========
    st.markdown("### Daily Forecast Summary")
    pred_cols = st.columns(3)
    
    for idx, (_, row) in enumerate(daily_stats.iterrows()):
        with pred_cols[idx]:
            day_name = pd.to_datetime(row["date"]).strftime("%A")
            avg_aqi = row["avg_aqi"]
            category, color = get_aqi_category(avg_aqi)
            
            st.markdown(f"""
                <div style='background: {color}22; border: 2px solid {color}; 
                            padding: 15px; border-radius: 10px; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9em; font-weight: bold;'>{day_name}</p>
                    <p style='margin: 0; font-size: 0.8em; color: #666;'>{row["date"]}</p>
                    <h2 style='margin: 10px 0; color: {color};'>{avg_aqi:.0f}</h2>
                    <p style='margin: 0; font-size: 0.85em;'>{category}</p>
                    <p style='margin: 5px 0 0 0; font-size: 0.75em; color: #999;'>
                        Range: {row["min_aqi"]:.0f} - {row["max_aqi"]:.0f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========= DETAILED CHARTS =========
    st.markdown("### Detailed Forecast Charts")
    
    chart_tabs = st.tabs(["Hourly Trend", "Daily Comparison", "Data Table"])
    
    with chart_tabs[0]:
        # Enhanced line chart with annotations
        st.write("**72-Hour Forecast**")
        st.line_chart(
            future_results.set_index("datetime")["predicted_aqi"],
            height=400
        )
        
        # Show peak hours
        peak_idx = future_results["predicted_aqi"].idxmax()
        peak_time = future_results.loc[peak_idx, "datetime"]
        peak_aqi = future_results.loc[peak_idx, "predicted_aqi"]
        
        low_idx = future_results["predicted_aqi"].idxmin()
        low_time = future_results.loc[low_idx, "datetime"]
        low_aqi = future_results.loc[low_idx, "predicted_aqi"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "🔺 Peak AQI", 
                f"{peak_aqi:.0f}",
                delta=f"at {pd.to_datetime(peak_time).strftime('%a %I:%M %p')}"
            )
        with col2:
            st.metric(
                "🔻 Lowest AQI",
                f"{low_aqi:.0f}",
                delta=f"at {pd.to_datetime(low_time).strftime('%a %I:%M %p')}"
            )
    
    with chart_tabs[1]:
        # Daily comparison bar chart
        st.write("**Daily Average Comparison**")
        st.bar_chart(daily_stats.set_index("date")["avg_aqi"], height=350)
        
        # Trend analysis
        if daily_stats["avg_aqi"].iloc[-1] < daily_stats["avg_aqi"].iloc[0]:
            st.success("Overall improving trend over the next 3 days")
        elif daily_stats["avg_aqi"].iloc[-1] > daily_stats["avg_aqi"].iloc[0]:
            st.warning("Overall worsening trend over the next 3 days")
        else:
            st.info("Stable air quality expected over the next 3 days")
    
    with chart_tabs[2]:
        # Data table with formatting
        st.write("**Complete Forecast Data**")
        
        # Prepare display dataframe
        display_df = future_results.copy()
        display_df["datetime"] = pd.to_datetime(display_df["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
        display_df["predicted_aqi"] = display_df["predicted_aqi"].round(1)
        display_df["category"] = display_df["predicted_aqi"].apply(
            lambda x: get_aqi_category(x)[0]
        )
        
        st.dataframe(
            display_df[["datetime", "predicted_aqi", "category"]].rename(columns={
                "datetime": "Time",
                "predicted_aqi": "AQI",
                "category": "Category"
            }),
            hide_index=True,
            use_container_width=True,
            height=400
        )
    
    st.markdown("---")
    
    # ========= FORECAST INTERPRETATION =========
    st.markdown("### Forecast Insights")
    
    avg_aqi_today = df.tail(24)["aqi"].mean() if "aqi" in df.columns and len(df) >= 24 else None
    avg_pred_aqi_next = future_results.head(24)["predicted_aqi"].mean()
    
    insight_cols = st.columns(3)
    
    with insight_cols[0]:
        if avg_aqi_today and avg_pred_aqi_next:
            change = avg_pred_aqi_next - avg_aqi_today
            if change > 5:
                st.warning(f"**+{change:.0f} AQI point increase** expected in next 24h")
            elif change < -5:
                st.success(f"**{abs(change):.0f} AQI point decrease** expected in next 24h")
            else:
                st.info(f"**Stable** air quality expected (±{abs(change):.0f} points)")
    
    with insight_cols[1]:
        # Count hours in each category
        good_hours = (future_results["predicted_aqi"] <= 50).sum()
        moderate_hours = ((future_results["predicted_aqi"] > 50) & 
                         (future_results["predicted_aqi"] <= 100)).sum()
        unhealthy_hours = (future_results["predicted_aqi"] > 100).sum()
        
        st.info(f"""
        **Forecast Breakdown:**
        - Good: {good_hours} hours
        - Moderate: {moderate_hours} hours
        - Unhealthy: {unhealthy_hours} hours
        """)
    
    with insight_cols[2]:
        # Best time for outdoor activities
        good_times = future_results[future_results["predicted_aqi"] <= 50]
        if len(good_times) > 0:
            best_window = good_times.iloc[0]["datetime"]
            st.success(f"""
            **Best Time for Outdoors:**
            {pd.to_datetime(best_window).strftime('%A %I:%M %p')}
            (AQI: {good_times.iloc[0]['predicted_aqi']:.0f})
            """)
        else:
            st.warning("No 'Good' air quality periods predicted")
    
else:
    st.warning("No predictions available. Please run `python src/generate_predictions.py` to generate forecasts.")
    st.info("Run the prediction script to see 3-day AQI forecasts with detailed insights.")



# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-size: 0.9em;'>
        <b>Karachi AQI Prediction Dashboard</b><br>
        Powered by Machine Learning • Hopsworks Feature Store • Open-Meteo API<br>
        
    </p>
    <p style='font-size: 0.8em; margin-top: 10px;'>
        Location: Karachi, Pakistan (24.86°N, 67.00°E)<br>
        Disclaimer: For informational purposes only. Always refer to official sources for health advisories.
    </p>
</div>
""", unsafe_allow_html=True)