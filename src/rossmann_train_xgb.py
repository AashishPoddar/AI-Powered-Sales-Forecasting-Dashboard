import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "rossmann_processed.csv"
OUTDIR = ROOT / "outputs" / "forecasts"
OUTDIR.mkdir(parents=True, exist_ok=True)

def prepare_df(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    agg = df.groupby("Date", as_index=False)["Sales"].sum().rename(columns={"Date":"ds","Sales":"y"})
    agg = agg.sort_values("ds").reset_index(drop=True)

    for lag in (1,2,3,7,14):
        agg[f"lag_{lag}"] = agg["y"].shift(lag).bfill()
    agg["rmean_7"] = agg["y"].rolling(7, min_periods=1).mean().shift(1).bfill()

    return agg

def train_and_forecast(df, horizon=14, num_boost_round=200):
    X = df.drop(columns=["ds","y"])
    y = df["y"].values
    feature_names = X.columns.tolist()

    dtrain = xgb.DMatrix(X, y, feature_names=feature_names)
    params = {"objective":"reg:squarederror", "eval_metric":"rmse"}
    model = xgb.train(params, dtrain, num_boost_round=int(num_boost_round))

    last = df.iloc[-1].copy()
    preds = []
    for _ in range(horizon):
        Xpred_df = pd.DataFrame([last.drop(labels=["ds","y"])], columns=feature_names)
        dp = xgb.DMatrix(Xpred_df, feature_names=feature_names)
        pred = float(model.predict(dp)[0])
        preds.append(pred)

        new = last.copy()
        new["y"] = pred
        # update lags
        new["lag_14"] = last.get("lag_13", last.get("lag_14", last["y"]))
        new["lag_7"]  = last.get("lag_6", last.get("lag_7", last["y"]))
        new["lag_3"]  = last.get("lag_2", last.get("lag_3", last["y"]))
        new["lag_2"]  = last.get("lag_1", last.get("lag_2", last["y"]))
        new["lag_1"]  = last["y"]
        new["rmean_7"] = np.mean([last["y"], last.get("lag_1", last["y"]), last.get("lag_2", last["y"])])
        last = new

    future_dates = pd.date_range(df["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_df = pd.DataFrame({"ds": future_dates, "yhat": preds})
    return model, forecast_df

def main(args):
    df = prepare_df(PROCESSED)
    model, forecast_df = train_and_forecast(df, horizon=int(args.horizon), num_boost_round=int(args.rounds))
    out = OUTDIR / f"rossmann_xgb_forecast_h{args.horizon}.csv"
    forecast_df.to_csv(out, index=False)
    print("Saved forecast:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", default=14, help="forecast horizon in days")
    parser.add_argument("--rounds", default=200, help="xgboost num_boost_round")
    args = parser.parse_args()
    main(args)
