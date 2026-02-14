"""Check the status of Hopsworks materialization job and verify data"""
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

def check_job_status(feature_group_name="qartzai", version=1):
    """Check the materialization job status and fetch sample data"""
    try:
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            raise ValueError("HOPSWORKS_API_KEY not found in environment variables.")
        
        print("🔗 Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        
        print(f"📦 Getting feature group: {feature_group_name} v{version}")
        feature_group = fs.get_feature_group(name=feature_group_name, version=version)
        
        # Get job state
        job_state = feature_group.materialization_job.get_state()
        
        print(f"\n{'='*60}")
        print(f"FEATURE GROUP: {feature_group_name} v{version}")
        print(f"{'='*60}")
        print(f"Job State: {job_state}")
        
        # Provide context based on state
        if job_state == "SUCCEEDED":
            print("✅ Job completed successfully!")
        elif job_state in ["INITIALIZING", "SUBMITTED", "RUNNING"]:
            print("⏳ Job is still running...")
        elif job_state == "FAILED":
            print("❌ Job failed!")
        elif job_state == "AGGREGATING_LOGS":
            print("📝 Job is finishing up (aggregating logs)...")
        
        # Try to read sample data
        print(f"\n{'='*60}")
        print("DATA VERIFICATION")
        print(f"{'='*60}")
        
        try:
            print("📥 Fetching sample data from feature group...")
            df = feature_group.read()
            
            print(f"\n✅ Data successfully retrieved!")
            print(f"📊 Total rows: {len(df)}")
            print(f"📋 Total columns: {len(df.columns)}")
            print(f"\n📝 Columns: {list(df.columns)}")
            
            if 'datetime' in df.columns:
                print(f"\n📅 Date range:")
                print(f"   Earliest: {df['datetime'].min()}")
                print(f"   Latest: {df['datetime'].max()}")
            
            if 'aqi' in df.columns:
                print(f"\n🌡️ AQI Statistics:")
                print(f"   Mean: {df['aqi'].mean():.2f}")
                print(f"   Min: {df['aqi'].min():.2f}")
                print(f"   Max: {df['aqi'].max():.2f}")
            
            print(f"\n📄 Sample data (first 5 rows):")
            print(df.head())
            
        except Exception as e:
            print(f"⚠️ Could not read data: {e}")
            print("Data might still be processing. Try again in a few minutes.")
        
        print(f"\n🔗 View in Hopsworks: https://c.app.hopsworks.ai/p/1336468/fs/1322063/fg/{feature_group.id}")
        
        return job_state
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        try:
            project.get_feature_store()._feature_store_api._close()
        except:
            pass

if __name__ == "__main__":
    check_job_status()
