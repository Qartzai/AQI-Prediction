import os, glob
import pandas as pd

def merge_all():
    hist = pd.read_csv("data/historical/historical_karachi_1y.csv")

    processed_files = glob.glob("data/processed/*.csv")
    latest = pd.concat([pd.read_csv(f) for f in processed_files]) if processed_files else pd.DataFrame()

    df = pd.concat([hist, latest])
    df.drop_duplicates(subset=["datetime"], inplace=True)
    df.sort_values("datetime", inplace=True)

    os.makedirs("data/final", exist_ok=True)
    out_file = "data/final/merged_karachi.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Final dataset saved → {out_file}")

if __name__ == "__main__":
    merge_all()