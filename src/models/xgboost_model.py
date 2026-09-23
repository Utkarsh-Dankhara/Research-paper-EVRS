"""
src/models/xgboost_model.py

Baseline Model 2 -- XGBoost (falls back to sklearn
GradientBoostingRegressor if the xgboost package isn't installed, same
as the original notebook's try/except). Ported from the "Model 2 --
Gradient Boosting (XGBoost)" section of
notebooks/EVRS_NN_multi_models.ipynb.

Fully self-contained: loads the cached sequence data itself, trains,
evaluates, and writes its own results -- does not depend on any other
model having run first or successfully.

Saves:
  models/xgboost/model.joblib
  results/xgboost/{metrics.json, predictions.npz, plots/eval_xgb.png}

Standalone: python -m src.models.xgboost_model
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

MODEL_NAME = "xgboost"
MODEL_PATH = config.MODELS_DIR / MODEL_NAME / "model.joblib"
RESULTS_DIR = config.RESULTS_DIR / MODEL_NAME


def _flattened_feature_names(feature_cols):
    n = len(feature_cols)
    return [
        f"{feature_cols[i % n]}[t-{config.SEQ_LEN - 1 - i // n}]"
        for i in range(config.SEQ_LEN * n)
    ]


def _train(X_flat_train, y_train, X_flat_val, y_val):
    """Trains XGBoost if available, otherwise falls back to sklearn's
    GradientBoostingRegressor (same hyperparameter philosophy: low
    learning rate + more trees, row/column subsampling for XGBoost).
    Returns (fitted_model, display_label)."""
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,      # low LR + more trees -- standard anti-overfit strategy
            subsample=0.8,           # row subsampling per tree
            colsample_bytree=0.8,    # feature subsampling per tree
            reg_alpha=0.1,           # L1 regularisation
            reg_lambda=1.5,          # L2 regularisation
            early_stopping_rounds=20,
            eval_metric="mae",
            random_state=config.SEED,
            n_jobs=-1,
            verbosity=0,
        )
        logger.info("Training XGBoost...")
        model.fit(X_flat_train, y_train, eval_set=[(X_flat_val, y_val)], verbose=False)
        return model, "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        logger.info("xgboost not installed -- falling back to sklearn GradientBoostingRegressor "
                    "(pip install xgboost to use the real thing).")
        model = GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=4, random_state=config.SEED,
        )
        model.fit(X_flat_train, y_train)
        return model, "Gradient Boosting"


def run(force: bool = False) -> dict:
    """
    Trains the XGBoost (or GradientBoosting fallback) baseline and
    evaluates it on the identical held-out test set used by every other
    model.

    Skips training (loads the existing model + metrics.json) if
    MODEL_PATH already exists and force=False.
    """
    if output_exists(MODEL_PATH) and not force:
        logger.info(f"Skipping training -- {MODEL_PATH} already exists (pass force=True to rerun).")
        if output_exists(RESULTS_DIR / "metrics.json"):
            return load_json(RESULTS_DIR / "metrics.json")
        return {}

    ensure_project_dirs()

    cache = load_sequence_cache()
    X_train, X_val, X_test = cache["X_train"], cache["X_val"], cache["X_test"]
    y_train, y_val, y_test = cache["y_train"], cache["y_val"], cache["y_test"]
    y_scaler = cache["y_scaler"]
    feature_cols = cache["feature_cols"]

    X_flat_train = X_train.reshape(len(X_train), -1)
    X_flat_val = X_val.reshape(len(X_val), -1)
    X_flat_test = X_test.reshape(len(X_test), -1)

    model, label = _train(X_flat_train, y_train, X_flat_val, y_val)

    pred_scaled = model.predict(X_flat_test)
    y_pred_real = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[:, 0]
    y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]

    metrics = regression_metrics(y_test_real, y_pred_real)
    metrics["backend"] = label
    logger.info(f"{label}: MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved -> {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS_DIR / "metrics.json", metrics)
    np.savez(RESULTS_DIR / "predictions.npz", y_true=y_test_real, y_pred=y_pred_real)

    if hasattr(model, "feature_importances_"):
        plot_tree_model_eval(
            y_test_real, y_pred_real,
            importances=model.feature_importances_,
            feature_names=_flattened_feature_names(feature_cols),
            model_label=f"{label} (risk_score_v1)",
            color=config.MODEL_COLORS[MODEL_NAME],
            save_path=RESULTS_DIR / "plots" / "eval_xgb.png",
        )
    else:
        # Defensive fallback -- every backend we actually use exposes
        # feature_importances_, but if a future backend doesn't, still
        # produce the scatter/residual half of the figure rather than
        # skipping the plot (and this model's results) entirely.
        from src.evaluation.plot_utils import plot_scatter_residual_only
        plot_scatter_residual_only(
            y_test_real, y_pred_real, f"{label} (risk_score_v1)",
            config.MODEL_COLORS[MODEL_NAME], RESULTS_DIR / "plots" / "eval_xgb.png",
        )

    return metrics


if __name__ == "__main__":
    from src.utils.cli import parse_force_flag
    from src.utils.logging_setup import get_logger
    get_logger()
    run(force=parse_force_flag())
