"""
Pipeline Orchestrator
---------------------
Orchestrates the Rossmann Sales Forecasting pipeline.
Runs preprocessing and model training/forecasting sequentially.

Usage:
  python run.py [--skip-preprocess] [--horizon 14] [--rounds 200] [--store <store_id>]
"""

import argparse
import sys
import time
from pathlib import Path
import subprocess
from typing import Optional

ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable  # Use current python environment


def run_pipeline(
    skip_preprocess: bool, horizon: int, rounds: int, store: Optional[int]
) -> None:
    """Execute preprocessing (if not skipped) followed by training."""
    start_time = time.time()
    
    print("=" * 60)
    print("Starting Rossmann Sales Forecasting Pipeline")
    print("=" * 60)

    # 1. Preprocessing
    if not skip_preprocess:
        print("\n--- Step 1: Preprocessing Raw Data ---")
        preprocess_script = ROOT / "src" / "rossmann_preprocess.py"
        
        t0 = time.time()
        res = subprocess.run([PYTHON_EXE, str(preprocess_script)], capture_output=False)
        if res.returncode != 0:
            print("[Error] Preprocessing failed.")
            sys.exit(res.returncode)
        print(f"Preprocessing completed in {time.time() - t0:.2f}s")
    else:
        print("\n--- Step 1: Skipping Preprocessing ---")

    # 2. Training and Forecasting
    print("\n--- Step 2: Training Model and Forecasting ---")
    train_script = ROOT / "src" / "rossmann_train_xgb.py"
    
    cmd = [
        PYTHON_EXE,
        str(train_script),
        "--horizon",
        str(horizon),
        "--rounds",
        str(rounds),
    ]
    if store is not None:
        cmd.extend(["--store", str(store)])

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=False)
    if res.returncode != 0:
        print("[Error] Training and forecasting failed.")
        sys.exit(res.returncode)
    print(f"Training and forecasting completed in {time.time() - t0:.2f}s")

    print("\n" + "=" * 60)
    print(f"Pipeline executed successfully in {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rossmann Sales Forecasting Pipeline Orchestrator"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip raw data preprocessing step.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Forecast horizon in days (default: 14).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="XGBoost num_boost_round (default: 200).",
    )
    parser.add_argument(
        "--store",
        type=int,
        default=None,
        help="Specific store ID to forecast. If omitted, aggregates globally.",
    )
    args = parser.parse_args()
    
    run_pipeline(
        skip_preprocess=args.skip_preprocess,
        horizon=args.horizon,
        rounds=args.rounds,
        store=args.store,
    )
