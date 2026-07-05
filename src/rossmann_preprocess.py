"""
Rossmann Sales Data Preprocessor
---------------------------------
Merges raw train and store data, handles missing values,
encodes categorical features, and engineers date-based features.

Input:  data/raw/rossmann_train.csv, data/raw/rossmann_store.csv
Output: data/processed/rossmann_processed.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def preprocess() -> None:
    """Load raw Rossmann data, clean it, engineer features, and save to CSV.

    Steps:
        1. Load train + store CSVs and merge on Store ID
        2. Fill missing competition / promo columns
        3. Encode categorical columns (StateHoliday, StoreType, Assortment)
        4. Create date-based features (Year, Month, Day, WeekOfYear, DayOfWeek)
        5. Filter to open-store rows only
        6. Save processed DataFrame to data/processed/
    """
    print("Loading datasets...")

    train = pd.read_csv(
        DATA_RAW / "rossmann_train.csv",
        parse_dates=["Date"],
        low_memory=False,
    )
    store = pd.read_csv(DATA_RAW / "rossmann_store.csv")

    print("Merging train + store...")
    df = train.merge(store, on="Store", how="left")

    print("Handling missing values...")
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)

    print("Encoding categorical columns...")
    pd.set_option("future.no_silent_downcasting", True)
    df["StateHoliday"] = (
        df["StateHoliday"]
        .replace({"0": 0, "a": 1, "b": 2, "c": 3})
        .astype(int)
    )
    df["StoreType"] = df["StoreType"].astype("category").cat.codes
    df["Assortment"] = df["Assortment"].astype("category").cat.codes

    print("Creating date features...")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Keep only open stores (closed stores have 0 sales, add noise)
    df = df[df["Open"] == 1]

    out_path = DATA_PROCESSED / "rossmann_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df):,} rows)")

if __name__ == "__main__":
    preprocess()
