"""
Script to verify Hopsworks Feature Store connection and data availability
"""
import os
import hopsworks
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    print("❌ HOPSWORKS_API_KEY not found in .env file")
    exit(1)

print("=" * 70)
print("🔍 Checking Hopsworks Feature Store")
print("=" * 70)

try:
    # Connect to Hopsworks
    print("\n1️⃣ Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    print(f"✅ Connected to project: {project.name}")
    
    # Get feature store
    print("\n2️⃣ Accessing Feature Store...")
    fs = project.get_feature_store(name='aqi_predictor_2')
    print(f"✅ Feature Store: {fs.name}")
    
    # List all feature groups
    print("\n3️⃣ Listing all Feature Groups...")
    all_fgs = fs.get_feature_groups()
    if all_fgs:
        print(f"📊 Found {len(all_fgs)} feature group(s):")
        for fg in all_fgs:
            print(f"   • {fg.name} (v{fg.version})")
    else:
        print("⚠️  No feature groups found!")
    
    # Check main feature group (qartzai_2)
    print("\n4️⃣ Checking main feature group: qartzai_2 v1")
    try:
        fg_main = fs.get_feature_group("qartzai_2", version=1)
        df_main = fg_main.read()
        print(f"✅ Feature Group: {fg_main.name} v{fg_main.version}")
        print(f"   📏 Shape: {df_main.shape} (rows, columns)")
        print(f"   📅 Columns: {list(df_main.columns)}")
        
        if "datetime" in df_main.columns:
            df_main["datetime"] = pd.to_datetime(df_main["datetime"])
            print(f"   ⏰ Time range: {df_main['datetime'].min()} to {df_main['datetime'].max()}")
        
        if "aqi" in df_main.columns:
            print(f"   🌫️  AQI range: {df_main['aqi'].min():.1f} - {df_main['aqi'].max():.1f}")
            print(f"   📊 Mean AQI: {df_main['aqi'].mean():.1f}")
        
        print(f"\n   Latest 5 rows:")
        if "datetime" in df_main.columns:
            print(df_main.sort_values("datetime", ascending=False).head(5)[["datetime", "aqi"]].to_string(index=False))
        else:
            print(df_main.head(5).to_string(index=False))
            
    except Exception as e:
        print(f"❌ Error accessing qartzai_2 v1: {e}")
    
    # Check predictions feature group
    print("\n5️⃣ Checking predictions feature group: qartzai_predictions v1")
    try:
        fg_pred = fs.get_feature_group("qartzai_predictions", version=1)
        df_pred = fg_pred.read()
        print(f"✅ Feature Group: {fg_pred.name} v{fg_pred.version}")
        print(f"   📏 Shape: {df_pred.shape} (rows, columns)")
        print(f"   📅 Columns: {list(df_pred.columns)}")
        
        if "datetime" in df_pred.columns:
            df_pred["datetime"] = pd.to_datetime(df_pred["datetime"])
            print(f"   ⏰ Time range: {df_pred['datetime'].min()} to {df_pred['datetime'].max()}")
        
        if "predicted_aqi" in df_pred.columns:
            print(f"   🔮 Predicted AQI range: {df_pred['predicted_aqi'].min():.1f} - {df_pred['predicted_aqi'].max():.1f}")
            print(f"   📊 Mean predicted AQI: {df_pred['predicted_aqi'].mean():.1f}")
        
        print(f"\n   Latest 5 predictions:")
        if "datetime" in df_pred.columns:
            print(df_pred.sort_values("datetime", ascending=False).head(5).to_string(index=False))
        else:
            print(df_pred.head(5).to_string(index=False))
            
    except Exception as e:
        print(f"❌ Error accessing qartzai_predictions v1: {e}")
    
    # Check Model Registry
    print("\n6️⃣ Checking Model Registry...")
    try:
        mr = project.get_model_registry()
        models = mr.get_models()
        if models:
            print(f"✅ Found {len(models)} model(s):")
            for model in models:
                print(f"   • {model.name} v{model.version}")
                
            # Try to get the specific model
            try:
                model = mr.get_model("aqi_prediction_model", version=3)
                print(f"\n   Current Model Details:")
                print(f"   📦 Name: {model.name} v{model.version}")
                print(f"   📝 Description: {model.description}")
            except:
                print(f"   ⚠️  aqi_prediction_model v3 not found")
        else:
            print("⚠️  No models found in registry!")
    except Exception as e:
        print(f"❌ Error accessing Model Registry: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Hopsworks check complete!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Fatal error: {e}")
    print("\nPossible issues:")
    print("  • Invalid API key")
    print("  • Network connection problem")
    print("  • Feature store or project deleted")
    print("  • Hopsworks service unavailable")
