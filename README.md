# 🌆 Karachi AQI Prediction Bot

![Feature Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/feature_pipeline.yml/badge.svg)
![Training Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/training_pipeline.yml/badge.svg)
![Prediction Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/prediction_pipeline.yml/badge.svg)
![PR Tests](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/pr_tests.yml/badge.svg)

> Real-time Air Quality Index predictions for Karachi using Machine Learning and automated CI/CD pipelines.

## 🎯 Overview

This project predicts Air Quality Index (AQI) for Karachi, Pakistan using machine learning models trained on 360 days of historical data. It provides:

- **Current AQI Status** - Both US EPA and European EAQI standards
- **3-Day Forecasts** - Hourly predictions with 96.3% accuracy
- **Health Recommendations** - Personalized advice based on air quality
- **Automated Updates** - Daily data collection and predictions via GitHub Actions

## ✨ Features

### 🌍 Dual AQI Standards
- **US EPA AQI** (0-500 scale) - North American standard
- **European EAQI** (0-300 scale) - European Environment Agency standard

### 🧪 Comprehensive Monitoring
- **6 Pollutants:** PM2.5, PM10, O₃, NO₂, SO₂, CO
- **4 Weather Metrics:** Temperature, Humidity, Wind Speed, Trend
- **24-Hour History:** Recent trends with statistics

### 📊 Advanced Forecasting
- **72-Hour Predictions** - Hourly granularity
- **Peak Time Detection** - When air quality is worst/best
- **Health Insights** - Best times for outdoor activities
- **Category Breakdown** - Hours in each air quality category

### 🤖 Automated CI/CD
- **Daily Data Collection** (3:00 AM UTC)
- **Weekly Model Training** (Sundays 3:30 AM UTC)
- **Daily Predictions** (4:00 AM UTC)
- **Pull Request Tests** (On every PR)

## 🚀 Quick Start

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

## 📁 Project Structure

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

## 🔄 CI/CD Pipeline

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

### Setup CI/CD

See [CICD_SETUP.md](CICD_SETUP.md) for detailed instructions.

**Quick Setup:**
1. Push code to GitHub
2. Add `HOPSWORKS_API_KEY` to GitHub Secrets
3. Enable workflows in Actions tab
4. Done! Pipelines run automatically

## 📊 Model Performance

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

## 🌐 Data Sources

- **Air Quality:** Open-Meteo Air Quality API
- **Weather:** Open-Meteo Forecast API
- **Location:** Karachi, Pakistan (24.86°N, 67.00°E)
- **Update Frequency:** Hourly observations, Daily forecasts

## 📖 Documentation

- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Complete usage guide
- [CICD_SETUP.md](CICD_SETUP.md) - GitHub Actions setup
- [FIXES_SUMMARY.md](FIXES_SUMMARY.md) - Bug fixes and improvements
- [DASHBOARD_ENHANCEMENTS.md](DASHBOARD_ENHANCEMENTS.md) - Dashboard features

## 🛠️ Development

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

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request (PR tests run automatically)

## 📈 Roadmap

- [ ] Add more cities (Lahore, Islamabad, Peshawar)
- [ ] SMS/Email alerts for unhealthy AQI
- [ ] Mobile app (React Native)
- [ ] Satellite imagery integration (NASA MODIS)
- [ ] Traffic data integration (Google Maps API)
- [ ] Historical comparison (year-over-year)
- [ ] API endpoint for external access
- [ ] Real-time streaming dashboard

## 🐛 Known Issues

- Hopsworks connection may timeout (uses local file fallback)
- European AQI calculated from simplified conversion
- Lag features less accurate when historical data unavailable

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- Open-Meteo for free air quality and weather data
- Hopsworks for feature store infrastructure
- Streamlit for dashboard framework
- scikit-learn for machine learning models

## 📞 Contact

- **Project Link:** https://github.com/YOUR_USERNAME/YOUR_REPO
- **Dashboard:** http://localhost:8501 (when running locally)

## ⚠️ Disclaimer

This project is for informational and educational purposes only. For official air quality advisories, please refer to government agencies and environmental authorities.

---

**Made with ❤️ for cleaner air in Karachi**

*Last updated: February 13, 2026*
