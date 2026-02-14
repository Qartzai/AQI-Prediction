import hopsworks
import pandas as pd
import os
from dotenv import load_dotenv
from joblib import load

load_dotenv()

def load_feature_data():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df["datetime"] = pd.to_datetime(df["datetime_str"])
    return df.sort_values("datetime")

def load_model():
    model_path = os.path.join("models", "best_model_random_forest.pkl")
    return load(model_path)