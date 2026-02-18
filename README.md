# 🌆 Karachi AQI Prediction Bot


> Real-time Air Quality Index predictions for Karachi using Machine Learning and automated CI/CD pipelines.

## Overview

This project predicts Air Quality Index (AQI) for Karachi, Pakistan using machine learning models trained on 360 days of historical data. It provides:

- **Current AQI Status** - Both US EPA and European EAQI standards
- **3-Day Forecasts** - Hourly predictions with 96.3% accuracy
- **Health Recommendations** - Personalized advice based on air quality
- **Automated Updates** - Daily data collection and predictions via GitHub Actions

## Features

### Dual AQI Standards
- **US EPA AQI** (0-500 scale) - North American standard
- **European EAQI** (0-300 scale) - European Environment Agency standard

### Comprehensive Monitoring
- **6 Pollutants:** PM2.5, PM10, O₃, NO₂, SO₂, CO
- **4 Weather Metrics:** Temperature, Humidity, Wind Speed, Trend
- **24-Hour History:** Recent trends with statistics

### Advanced Forecasting
- **72-Hour Predictions** - Hourly granularity
- **Peak Time Detection** - When air quality is worst/best
- **Health Insights** - Best times for outdoor activities
- **Category Breakdown** - Hours in each air quality category

### Automated CI/CD
- **Daily Data Collection** (3:00 AM UTC)
- **Weekly Model Training** (Sundays 3:30 AM UTC)
- **Daily Predictions** (4:00 AM UTC)
- **Pull Request Tests** (On every PR)

## Quick Start

### Prerequisites
```bash
Python 3.10+
pip
git
```

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
# Create .env file
echo "HOPSWORKS_API_KEY=your_api_key_here" > .env
echo "SAVE_LOCAL=false" >> .env
```

### Run Locally

1. **Get Latest Data**
```bash
python src/run_feature_pipeline.py
```

2. **Generate Predictions**
```bash
python src/generate_predictions.py
```

3. **Launch Dashboard**
```bash
streamlit run streamlit_app/app.py
```

4. **Open Browser**
```
http://localhost:8501
```

## Project Structure

```
NLP-Project/
├── .github/workflows/       # CI/CD pipelines
│   ├── feature_pipeline.yml
│   ├── training_pipeline.yml
│   ├── prediction_pipeline.yml
│   ├── pr_tests.yml
│   └── backfill.yml
├── src/                     # Source code
│   ├── config.py           # Configuration
│   ├── fetch_data.py       # API fetching
│   ├── process_data.py     # Data processing
│   ├── clean_data.py       # Data cleaning
│   ├── process_features.py # Feature engineering
│   ├── aqi_utils.py        # AQI calculations
│   ├── train_model.py      # Model training
│   ├── generate_predictions.py  # Prediction generation
│   ├── backfill_data.py    # Historical backfill
│   └── upload_hopsworks.py # Hopsworks upload
├── streamlit_app/          # Dashboard
│   └── app.py
├── data/                   # Data storage
│   ├── historical/         # 360-day historical data
│   ├── predictions/        # Generated forecasts
│   └── processed/          # Processed data
├── models/                 # Trained models
│   ├── best_model_*.pkl
│   └── scaler.pkl
└── notebooks/             # Jupyter notebooks
    ├── 01_eda_preprocessing.ipynb
    └── 02_eda_feature_analysis.ipynb
```

## CI/CD Pipeline

### Automated Workflows

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| **Feature Pipeline** | Daily 3:00 AM UTC | Fetch latest 24h AQI data |
| **Training Pipeline** | Weekly Sun 3:30 AM UTC | Retrain ML models |
| **Prediction Pipeline** | Daily 4:00 AM UTC | Generate 3-day forecasts |

### Manual Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **Backfill** | Manual | Refresh 360 days historical data |
| **PR Tests** | Pull Request | Code quality validation |


**Quick Setup:**
1. Push code to GitHub
2. Add `HOPSWORKS_API_KEY` to GitHub Secrets
3. Enable workflows in Actions tab
4. Done! Pipelines run automatically

## Model Performance

### Best Model: Random Forest
- **R² Score:** 0.963 (Excellent)
- **RMSE:** 11.5 AQI points
- **MAE:** 4.3 AQI points
- **Training Data:** 360 days (8,640 hours)

### Features Used
- **Pollutants:** PM2.5, PM10, O₃, NO₂, SO₂, CO
- **Weather:** Temperature, Humidity, Wind Speed/Direction
- **Time-based:** Hour, Day, Month, Weekday, Cyclic encoding
- **Derived:** PM ratio, Temp/Humidity ratio, Wind effect

## Data Sources

- **Air Quality:** Open-Meteo Air Quality API
- **Weather:** Open-Meteo Forecast API
- **Location:** Karachi, Pakistan (24.86°N, 67.00°E)
- **Update Frequency:** Hourly observations, Daily forecasts

## Development

### Run Tests
```bash
python -m pytest tests/
```

### Lint Code
```bash
flake8 src/ --max-line-length=127
```

### Format Code
```bash
black src/ streamlit_app/
```

### Update Dependencies
```bash
pip freeze > requirements.txt
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request (PR tests run automatically)

## Roadmap

- [ ] Add more cities (Lahore, Islamabad, Peshawar)
- [ ] SMS/Email alerts for unhealthy AQI
- [ ] Mobile app (React Native)
- [ ] Satellite imagery integration (NASA MODIS)
- [ ] Traffic data integration (Google Maps API)
- [ ] Historical comparison (year-over-year)
- [ ] API endpoint for external access
- [ ] Real-time streaming dashboard

## Known Issues

- Hopsworks connection may timeout (uses local file fallback)
- European AQI calculated from simplified conversion
- Lag features less accurate when historical data unavailable

## Troubleshooting & Lessons Learned

During development, we ran into several issues that might help others working on similar projects. Here's what went wrong and how we fixed it:

### 1. Dashboard Showing Wrong Data
**Problem:** The dashboard was displaying AQI values of 500 (which is basically toxic air) and predictions were stuck at 95 for all three days. Pollutants like CO were showing 283.8 ppm instead of the actual 1.0 ppm.

**Fix:** Turned out the stored AQI values couldn't be trusted because of incorrect unit conversions. We added a function to recalculate AQI from raw pollutant data every time, using proper molecular weight conversions (µg/m³ to ppb/ppm). Now the dashboard shows accurate real-time values.

### 2. Weird 500 AQI Spike in Charts
**Problem:** The 24-hour chart would randomly spike to 500 at the last data point, making it look like sudden apocalyptic air quality.

**Fix:** Same root cause as above - one stored AQI value was calculated with wrong units. Instead of fixing that one value, we made the chart recalculate all AQI values from raw data. Problem disappeared.

### 3. Category Display Crashing
**Problem:** The app would crash with `IndexError` when trying to display AQI categories because we were calling `.split()[1]` on category names.

**Fix:** Removed the `.split()[1]` entirely since the `get_aqi_category()` function already returns clean category names like "Good" or "Moderate". No splitting needed.

### 4. GitHub Actions Prediction Workflow Failing
**Problem:** The daily prediction workflow kept failing with `KeyError: "['datetime'] not in index"`. Super frustrating because it would work locally but fail on GitHub.

**Fix:** The issue was inconsistent column naming - sometimes Hopsworks stored it as "datetime" and sometimes as "datetime_str". We updated the code to handle both cases defensively.

### 5. The Big Datetime Mess
**Problem:** Everything broke after we discovered that datetime values were being converted to strings throughout the entire codebase. This was causing schema mismatches and the KeyError above.

**Fix:** This required a bigger fix across 6 files:
- Changed `upload_hopsworks.py` to keep datetime as an actual datetime type instead of converting to string
- Updated primary_key from `["datetime_str"]` to `["datetime"]`
- Added `event_time="datetime"` parameter to properly tell Hopsworks this is a timestamp
- Fixed all dependent files to expect datetime columns, not string columns

**Lesson:** Don't convert datetime columns to strings unless you absolutely have to. Databases and feature stores need proper timestamp types for time-based operations.

### 6. Upload Function Eating the Datetime Column
**Problem:** After "fixing" datetime handling, predictions started failing again because the datetime column would disappear after uploading to Hopsworks.

**Fix:** The upload function was modifying the original DataFrame. We changed it to `.copy()` the DataFrame before uploading, keeping the original intact for display later.

### 7. Streamlit Cloud Stuck on Loading Screen
**Problem:** The app deployed to Streamlit Cloud but would just show a loading spinner forever. Locally it worked fine.

**Fix:** Three issues:
- Initially removed `python-dotenv` from requirements (Streamlit Cloud doesn't need it)
- Changed API key loading to try `st.secrets` first for cloud deployment
- Added proper error handling so the app wouldn't crash when local files don't exist in cloud environment

### 8. Local Streamlit Not Reading .env File
**Problem:** After "fixing" cloud deployment, local development broke because the app couldn't find the `HOPSWORKS_API_KEY` from the `.env` file.

**Fix:** 
- Re-added `python-dotenv` to requirements with try-except import (optional dependency)
- Specified explicit path to `.env` file: `os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')`
- Changed priority to check environment variables first, then Streamlit secrets

**Lesson:** When supporting both local and cloud deployment, always specify explicit file paths and test both environments separately.

### 9. Requirements.txt Breaking on Streamlit Cloud
**Problem:** Streamlit Cloud deployment failed with "Error installing requirements" even though packages were correct.

**Fix:** Streamlit Cloud is picky about `requirements.txt` format:
- Removed all comments (lines starting with #)
- Removed duplicate package entries (`hopsworks` was listed twice)
- Removed `confluent-kafka` which wasn't actually needed
- Removed version pinning to let it install compatible versions

**Takeaway:** Most of the issues came down to:
1. **Trust but verify** - Don't assume stored calculations are correct; recalculate when displaying critical data
2. **Type consistency** - Keep datetime as datetime, not strings
3. **Defensive coding** - Handle both old and new column names during transitions
4. **Environment differences** - What works locally might not work in cloud; test both
5. **Clean dependencies** - Only include what you actually use

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Authors

- **Your Name** - Initial work

## Acknowledgments

- Open-Meteo for free air quality and weather data
- Hopsworks for feature store infrastructure
- Streamlit for dashboard framework
- scikit-learn for machine learning models

## Contact

- **Project Link:** https://github.com/YOUR_USERNAME/YOUR_REPO
- **Dashboard:** http://localhost:8501 (when running locally)

## Disclaimer

This project is for informational and educational purposes only. For official air quality advisories, please refer to government agencies and environmental authorities.

---

