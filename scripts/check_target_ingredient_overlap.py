"""
scripts/check_target_ingredient_overlap.py
--------------------------------------------
Diagnostic: how much of risk_score_v2's variance can be explained using
ONLY its raw shared ingredient columns -- speed_kph, junction_count,
sinuosity, length, betweenness_centrality -- the same columns also given
to every model as raw input features, with NO sequence context and NO
other features.

CORRECTED: an earlier pass at this analysis (in conversation, not in
this file) claimed betweenness_centrality was withheld from the model's
features, the one part of risk_score_v2's formula the models never see.
That was wrong -- config.py's CANDIDATE_FEATURE_COLS includes BOTH
betweenness_centrality AND betweenness_centrality_pct_in_city directly.
So there is no withheld raw ingredient at all: all 5 of risk_score_v2's
structural components are directly available to every model as features.
This script tests all 5 (plus the pre-computed _pct_in_city version,
since it's a separate model feature and likely built via a similar
per-city percentile-rank methodology to what risk_v2.py does internally
for its own betweenness term).

Why this matters: risk_score_v2 is built (src/data/risk_v2.py) from a
per-city percentile rank of these components and a speed x junction
interaction term. If a bare model using only these raw/pct columns
already gets close to the full pipeline's R^2 (XGBoost 0.9925, Bayesian
BiLSTM ~0.9929), that's evidence the 5-step sequence context and the
remaining features aren't adding much -- the models are mostly learning
to approximate the rank-transform of a handful of columns they're
already directly handed, not something genuinely richer. If it lands
meaningfully lower, that's evidence the full feature set and sequence
context are contributing real signal beyond these ingredient columns.

This does NOT reproduce the exact same held-out rows the main pipeline
used (that split happens after sequence-building, on a row_meta.csv that
maps sequence rows back to specific roads) -- it's an independent split
with the same seed/test_size on the processed CSV directly. That's a
deliberate simplification: this test only needs to be directionally
correct, not byte-for-byte identical to the main pipeline's split.

Usage:
    python scripts/check_target_ingredient_overlap.py
    python scripts/check_target_ingredient_overlap.py --csv path/to/roads_all_cities_processed.csv

Edit FULL_PIPELINE_REFERENCE below if your comparison numbers change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# CORRECTED: all 5 of risk_score_v2's raw structural components are
# directly in CANDIDATE_FEATURE_COLS -- none are withheld. Testing both
# the raw betweenness_centrality and its pre-computed percentile version,
# since both are separately available to every model as features.
RAW_INGREDIENT_COLS = [
    "speed_kph", "junction_count", "sinuosity", "length",
    "betweenness_centrality", "betweenness_centrality_pct_in_city",
]
TARGET_COL = "risk_score_v2"
SEED = 42
TEST_SIZE = 0.15

# For context only -- what the FULL pipeline (22 features, 5-step sequence
# context) achieved on your last comparison run. Edit these if they change.
FULL_PIPELINE_REFERENCE = {
    "Random Forest (full pipeline)":     {"mae": 0.0248, "rmse": 0.0329, "r2": 0.9175},
    "XGBoost (full pipeline)":           {"mae": 0.0073, "rmse": 0.0099, "r2": 0.9925},
    "Vanilla LSTM (full pipeline)":      {"mae": 0.0130, "rmse": 0.0177, "r2": 0.9761},
    "Transformer (full pipeline)":       {"mae": 0.0170, "rmse": 0.0209, "r2": 0.9669},
    "Bayesian BiLSTM (full pipeline)":   {"mae": 0.0046, "rmse": 0.0097, "r2": 0.9929},
}


def evaluate(name, model, X_train, y_train, X_test, y_test, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"mae": mae, "rmse": rmse, "r2": r2}
    print(f"{name:45s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--csv", type=str,
        default="data/processed/roads_all_cities_processed.csv",
        help="Path to the processed roads CSV (default: %(default)s)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"Pass the correct path, e.g.:")
        print(f"  python {Path(__file__).name} --csv /path/to/roads_all_cities_processed.csv")
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    missing = [c for c in RAW_INGREDIENT_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print(f"ERROR: missing expected columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=RAW_INGREDIENT_COLS + [TARGET_COL]).reset_index(drop=True)
    print(f"{len(df)} rows after dropping missing values in the relevant columns "
          f"({before - len(df)} dropped).\n")

    X = df[RAW_INGREDIENT_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print("(independent split, same seed/test_size as the main pipeline -- not the")
    print(" exact same held-out rows, but methodologically comparable)\n")

    print("=" * 90)
    print(f"BARE-BONES MODELS -- using ONLY {RAW_INGREDIENT_COLS}")
    print("(no sequence context, no other features -- these ARE the columns")
    print(" risk_score_v2 is built from, and all of them are also given to")
    print(" every model directly as input features)")
    print("=" * 90)

    results: dict = {}
    evaluate(
        "Linear Regression (4 raw columns only)",
        LinearRegression(), X_train, y_train, X_test, y_test, results,
    )
    evaluate(
        "Random Forest, shallow (4 raw columns only)",
        RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=4,
                               random_state=SEED, n_jobs=-1),
        X_train, y_train, X_test, y_test, results,
    )
    evaluate(
        "Random Forest, deep (4 raw columns only)",
        RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=2,
                               random_state=SEED, n_jobs=-1),
        X_train, y_train, X_test, y_test, results,
    )

    print()
    print("=" * 90)
    print("FOR REFERENCE -- full pipeline (22 features, 5-step sequence context)")
    print("=" * 90)
    for name, m in FULL_PIPELINE_REFERENCE.items():
        print(f"{name:45s}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")

    best_bare_r2 = max(r["r2"] for r in results.values())
    best_full_r2 = max(m["r2"] for m in FULL_PIPELINE_REFERENCE.values())
    gap = best_full_r2 - best_bare_r2

    print()
    print("=" * 90)
    print("HOW TO READ THIS")
    print("=" * 90)
    print(f"Best bare-bones R^2 (4 raw columns only): {best_bare_r2:.4f}")
    print(f"Best full-pipeline R^2 (22 features + sequence): {best_full_r2:.4f}")
    print(f"Gap: {gap:.4f}\n")

    if gap < 0.03:
        print("-> The 4 raw ingredient columns alone get NEARLY as far as the full")
        print("   22-feature, 5-step-sequence pipeline. This suggests most of the")
        print("   models' predictive power comes from approximating the rank-transform")
        print("   of these 4 columns, not from the sequence context or the other 18")
        print("   features. Worth reporting as a real limitation of the evaluation.")
    elif gap < 0.10:
        print("-> Moderate gap. The full pipeline does meaningfully better than the")
        print("   raw ingredients alone, but the bare-bones model still gets quite")
        print("   far -- some real signal is coming from elsewhere, but the raw")
        print("   ingredients carry a lot of the weight.")
    else:
        print("-> Large gap. The full pipeline's sequence context and additional")
        print("   features are clearly contributing substantial signal beyond what")
        print("   the 4 raw ingredient columns alone can explain -- good evidence")
        print("   the models are learning something beyond just the rank-transform")
        print("   of a few columns.")


if __name__ == "__main__":
    main()
