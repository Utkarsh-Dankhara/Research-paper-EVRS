"""
src/models/random_forest.py

Baseline Model 1 -- RandomForestRegressor on flattened sequences.
Ported from the "Model 1 -- Random Forest Regressor" section of
notebooks/EVRS_NN_multi_models.ipynb (n_estimators=500, max_depth=12,
min_samples_leaf=4, max_features='sqrt').

Fully self-contained: loads the cached sequence data itself, trains,
evaluates, and writes its own results -- does not depend on any other
model having run first or successfully.

Saves:
  models/random_forest/model.joblib
  results/random_forest/{metrics.json, predictions.npz, plots/eval_rf.png}

Standalone: python -m src.models.random_forest
"""
import logging

import joblib
import numpy as np

import config
from src.data.loaders import load_sequence_cache
from src.evaluation.metrics import regression_metrics
from src.evaluation.plot_utils import plot_tree_model_eval
from src.utils.io import ensure_project_dirs, load_json, output_exists, save_json

logger = logging.getLogger(__name__)

MODEL_NAME = "random_forest"
MODEL_PATH = config.MODELS_DIR / MODEL_NAME / "model.joblib"
RESULTS_DIR = config.RESULTS_DIR / MODEL_NAME


def _flattened_feature_names(feature_cols):
    """Expands feature names across the SEQ_LEN steps for feature-
    importance plot labels, e.g. 'length[t-4]' ... 'length[t-0]'."""
    n = len(feature_cols)
    return [
        f"{feature_cols[i % n]}[t-{config.SEQ_LEN - 1 - i // n}]"
        for i in range(config.SEQ_LEN * n)
    ]


def run(force: bool = False) -> dict:
    """
    Trains the Random Forest baseline and evaluates it on the identical
    held-out test set used by every other model.

    Skips training (loads the existing model + metrics.json) if
    MODEL_PATH already exists and force=False.
    """
    if output_exists(MODEL_PATH) and not force:
        logger.info(f"Skipping training -- {MODEL_PATH} already exists (pass force=True to rerun).")
        if output_exists(RESULTS_DIR / "metrics.json"):
            return load_json(RESULTS_DIR / "metrics.json")
        return {}

    from sklearn.ensemble import RandomForestRegressor

    ensure_project_dirs()

    cache = load_sequence_cache()
    X_train, X_test = cache["X_train"], cache["X_test"]
    y_train, y_test = cache["y_train"], cache["y_test"]
    y_scaler = cache["y_scaler"]
    feature_cols = cache["feature_cols"]

    # Flat view for tree-based models (N x SEQ_LEN*n_features)
    X_flat_train = X_train.reshape(len(X_train), -1)
    X_flat_test = X_test.reshape(len(X_test), -1)

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,          # limits tree depth -> prevents overfitting on a few thousand samples
        min_samples_leaf=4,    # requires at least 4 samples per leaf -> smoother predictions
        max_features="sqrt",   # sqrt(n_features) per split -- standard RF variance reduction
        n_jobs=-1,
        random_state=config.SEED,
    )
    logger.info("Training Random Forest (500 trees)...")
    rf.fit(X_flat_train, y_train)

    pred_scaled = rf.predict(X_flat_test)
    y_pred_real = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[:, 0]
    y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]

    metrics = regression_metrics(y_test_real, y_pred_real)
    logger.info(f"Random Forest: MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    logger.info(f"Model saved -> {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS_DIR / "metrics.json", metrics)
    np.savez(RESULTS_DIR / "predictions.npz", y_true=y_test_real, y_pred=y_pred_real)

    plot_tree_model_eval(
        y_test_real, y_pred_real,
        importances=rf.feature_importances_,
        feature_names=_flattened_feature_names(feature_cols),
        model_label=config.MODEL_LABELS[MODEL_NAME],
        color=config.MODEL_COLORS[MODEL_NAME],
        save_path=RESULTS_DIR / "plots" / "eval_rf.png",
    )

    return metrics


if __name__ == "__main__":
    from src.utils.cli import parse_force_flag
    from src.utils.logging_setup import get_logger
    get_logger()
    run(force=parse_force_flag())
