"""
Rossmann XGBoost Sales Forecaster
----------------------------------
Trains an XGBoost regressor on aggregated daily sales with lag/rolling
features, evaluates on a held-out period, and produces a multi-step forecast.
Supports global forecasting or specific store-level forecasting.

Input:  data/processed/rossmann_processed.csv
Output: outputs/forecasts/rossmann_xgb_forecast_h{horizon}.csv
        outputs/forecasts/rossmann_xgb_forecast_h{horizon}_pbi.csv
        outputs/models/rossmann_xgb_model.json
        outputs/plots/feature_importance.png
        outputs/plots/val_actual_vs_pred.png
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Headless matplotlib config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "rossmann_processed.csv"
OUTDIR = ROOT / "outputs" / "forecasts"
MODELDIR = ROOT / "outputs" / "models"
PLOTDIR = ROOT / "outputs" / "plots"

OUTDIR.mkdir(parents=True, exist_ok=True)
MODELDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

# ---------- feature config ----------
LAG_DAYS: List[int] = [1, 2, 3, 7, 14]
ROLLING_WINDOW: int = 7


def prepare_df(path: Path, store_id: Optional[int] = None) -> pd.DataFrame:
    """Load processed CSV and build lag + rolling-mean features.
    Optionally filters data for a specific store.

    Args:
        path: Path to the processed Rossmann CSV.
        store_id: Optional Store ID to filter. If None, aggregates globally.

    Returns:
        DataFrame with columns: ds (date), y (total daily sales),
        lag_1 … lag_14, rmean_7.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    
    if store_id is not None:
        print(f"Filtering data for Store: {store_id}")
        df = df[df["Store"] == store_id]
    else:
        print("Aggregating sales globally across all stores")

    agg = (
        df.groupby("Date", as_index=False)["Sales"]
        .sum()
        .rename(columns={"Date": "ds", "Sales": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )

    # Lag features
    for lag in LAG_DAYS:
        agg[f"lag_{lag}"] = agg["y"].shift(lag)

    # Rolling mean (shifted by 1 to avoid data leakage)
    agg[f"rmean_{ROLLING_WINDOW}"] = (
        agg["y"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .mean()
        .shift(1)
    )

    # Drop rows where lag features are NaN (first 14 rows)
    agg = agg.dropna().reset_index(drop=True)

    return agg


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature column names (everything except ds and y)."""
    return [c for c in df.columns if c not in ("ds", "y")]


def save_plots(
    model: xgb.Booster,
    y_val: np.ndarray,
    val_preds: np.ndarray,
    dates: pd.Series,
    store_id: Optional[int],
) -> None:
    """Generate and save evaluation plots (feature importance and actual vs predicted)."""
    # 1. Feature Importance Plot
    importance = model.get_score(importance_type="gain")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=False)
    
    if sorted_importance:
        features, scores = zip(*sorted_importance)
        plt.figure(figsize=(10, 6))
        plt.barh(features, scores, color="#2b5c8f")
        plt.title(f"XGBoost Feature Importance (Gain) - Store {store_id if store_id else 'Global'}")
        plt.xlabel("Gain")
        plt.tight_layout()
        plot_path = PLOTDIR / "feature_importance.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved feature importance plot: {plot_path}")

    # 2. Actual vs Predicted Validation Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_val, label="Actual Sales", marker="o", color="#2b5c8f", linewidth=2)
    plt.plot(dates, val_preds, label="Predicted Sales", marker="x", color="#ff7f0e", linestyle="--", linewidth=2)
    plt.title(f"Validation Period: Actual vs Predicted Sales - Store {store_id if store_id else 'Global'}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    val_plot_path = PLOTDIR / "val_actual_vs_pred.png"
    plt.savefig(val_plot_path, dpi=150)
    plt.close()
    print(f"Saved validation actual vs predicted plot: {val_plot_path}")


def train_and_forecast(
    df: pd.DataFrame,
    horizon: int = 14,
    num_boost_round: int = 200,
    eval_days: int = 14,
    store_id: Optional[int] = None,
) -> Tuple[xgb.Booster, pd.DataFrame, dict]:
    """Train XGBoost, evaluate on a holdout set, and generate future forecast.

    Args:
        df: Prepared DataFrame from prepare_df().
        horizon: Number of days to forecast into the future.
        num_boost_round: XGBoost boosting rounds.
        eval_days: Number of trailing days to hold out for evaluation.
        store_id: Optional store ID for plotting metadata.

    Returns:
        (model, forecast_df, metrics_dict)
        - model: trained xgb.Booster
        - forecast_df: DataFrame with 'ds' and 'yhat' columns
        - metrics_dict: dict with RMSE, MAE, MAPE on the holdout set
    """
    feature_cols = get_feature_cols(df)

    # ---- Train / Validation Split ----
    train_df = df.iloc[:-eval_days]
    val_df = df.iloc[-eval_days:]

    X_train = train_df[feature_cols]
    y_train = train_df["y"].values
    X_val = val_df[feature_cols]
    y_val = val_df["y"].values

    dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, y_val, feature_names=feature_cols)

    params = {"objective": "reg:squarederror", "eval_metric": "rmse"}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=50,
    )

    # ---- Evaluation Metrics ----
    val_preds = model.predict(dval)
    rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    mae = float(mean_absolute_error(y_val, val_preds))
    mape = float(np.mean(np.abs((y_val - val_preds) / y_val)) * 100)
    metrics = {"RMSE": rmse, "MAE": mae, "MAPE(%)": mape}

    print("\n--- Validation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:,.2f}")
    print()

    # Save evaluation plots
    save_plots(model, y_val, val_preds, val_df["ds"], store_id)

    # ---- Retrain on full data for final forecast ----
    X_full = df[feature_cols]
    y_full = df["y"].values
    dfull = xgb.DMatrix(X_full, y_full, feature_names=feature_cols)
    model = xgb.train(params, dfull, num_boost_round=int(num_boost_round))

    # ---- Multi-step Forecast ----
    forecast_df = _recursive_forecast(model, df, feature_cols, horizon)

    return model, forecast_df, metrics


def _recursive_forecast(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
) -> pd.DataFrame:
    """Generate multi-step ahead forecast by recursively updating lag features.

    At each step the predicted value is fed back as lag_1, the previous lag_1
    becomes lag_2, etc.  The rolling mean is recomputed from the most recent
    ROLLING_WINDOW values.

    Args:
        model: Trained XGBoost Booster.
        df: Full prepared DataFrame (used to seed the history buffer).
        feature_cols: Ordered list of feature column names.
        horizon: Number of future days to predict.

    Returns:
        DataFrame with 'ds' (future dates) and 'yhat' (predictions).
    """
    # Keep a history buffer of recent y values for lag/rolling computation
    history = df["y"].tolist()

    preds: List[float] = []
    last_date = df["ds"].iloc[-1]

    for step in range(horizon):
        row = {}

        # Build lag features from history
        for lag in LAG_DAYS:
            if len(history) >= lag:
                row[f"lag_{lag}"] = history[-lag]
            else:
                row[f"lag_{lag}"] = history[0]  # fallback for very short series

        # Rolling mean of last ROLLING_WINDOW values
        window = history[-ROLLING_WINDOW:]
        row[f"rmean_{ROLLING_WINDOW}"] = float(np.mean(window))

        # Predict
        Xpred = pd.DataFrame([row], columns=feature_cols)
        dp = xgb.DMatrix(Xpred, feature_names=feature_cols)
        pred = float(model.predict(dp)[0])
        preds.append(pred)

        # Append prediction to history so next step can use it as a lag
        history.append(pred)

    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    return pd.DataFrame({"ds": future_dates, "yhat": preds})


def main(args: argparse.Namespace) -> None:
    """Entry point: prepare data, train, forecast, save model, and save outputs."""
    df = prepare_df(PROCESSED, store_id=args.store)

    horizon = int(args.horizon)
    rounds = int(args.rounds)

    model, forecast_df, metrics = train_and_forecast(
        df, horizon=horizon, num_boost_round=rounds, store_id=args.store
    )

    # Save standard forecast CSV
    out = OUTDIR / f"rossmann_xgb_forecast_h{horizon}.csv"
    forecast_df.to_csv(out, index=False)
    print(f"Saved forecast: {out}")

    # Auto-generate Power BI format (Date, Forecast columns)
    pbi_df = forecast_df.rename(columns={"ds": "Date", "yhat": "Forecast"})
    out_pbi = OUTDIR / f"rossmann_xgb_forecast_h{horizon}_pbi.csv"
    pbi_df.to_csv(out_pbi, index=False)
    print(f"Saved PBI forecast: {out_pbi}")

    # Save trained model to JSON
    model_out = MODELDIR / "rossmann_xgb_model.json"
    model.save_model(str(model_out))
    print(f"Saved model to: {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost on Rossmann sales and generate forecast"
    )
    parser.add_argument(
        "--horizon", type=int, default=14,
        help="Forecast horizon in days (default: 14)"
    )
    parser.add_argument(
        "--rounds", type=int, default=200,
        help="XGBoost num_boost_round (default: 200)"
    )
    parser.add_argument(
        "--store", type=int, default=None,
        help="Specific store ID to forecast. If omitted, aggregates all stores globally."
    )
    args = parser.parse_args()
    main(args)
