import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

def preprocess():
    
    print("Loading datasets...")

    train = pd.read_csv(
        DATA_RAW / "rossmann_train.csv",
        parse_dates=["Date"]
    )

    store = pd.read_csv(DATA_RAW / "rossmann_store.csv")
    test = pd.read_csv(DATA_RAW / "rossmann_test.csv", parse_dates=["Date"])

    print("Merging train + store...")
    df = train.merge(store, on="Store", how="left")

    print("Handling missing values...")
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)

    print("Encoding categorical columns...")
    df["StateHoliday"] = df["StateHoliday"].replace({"0": 0, "a": 1, "b": 2, "c": 3}).astype(int)
    df["StoreType"] = df["StoreType"].astype("category").cat.codes
    df["Assortment"] = df["Assortment"].astype("category").cat.codes

    print("Creating date features...")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Keep only open stores
    df = df[df["Open"] == 1]

    out_path = DATA_PROCESSED / "rossmann_processed.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    preprocess()
