"""
src/models/base.py

Shared pieces every model module builds on, so main.py can loop over all
5 models identically and each stays independently runnable:

  - compile_and_train_keras(...) -- the identical optimizer/loss/callback
    setup the original notebook used for all 3 Keras-based models
    (Bayesian BiLSTM, vanilla LSTM, Transformer), centralized so it only
    has to be gotten right once.
  - run_model_stage(...) -- wraps one model's run() in try/except so a
    single failure is logged and skipped instead of stopping the other
    4 models. Used by main.py's training loop.

Each model module (bayesian_bilstm.py, random_forest.py, xgboost_model.py,
vanilla_lstm.py, transformer.py) exposes the same shape:

  run(force: bool = False) -> dict
      Self-contained entry point: loads the cached sequence data itself
      (via src.data.loaders), trains, evaluates, and writes
      results/<model_name>/{metrics.json, predictions.npz, plots/...}
      plus the model artifact under models/<model_name>/. Must not
      depend on any other model module's run(). Returns the metrics dict.
      Skips training (loads existing artifact + metrics) if the model
      artifact already exists and force=False.

  Also runnable standalone, e.g.:
      python -m src.models.transformer
"""
import logging

import config

logger = logging.getLogger(__name__)


def compile_and_train_keras(model, X_train, y_train, X_val, y_val,
                             learning_rate=5e-4, early_stop_patience=None,
                             plateau_patience=None, verbose=1):
    """Shared training loop for every Keras-based model. Same
    optimizer/loss/callbacks the original notebook used for the Bayesian
    BiLSTM, vanilla LSTM, and Transformer, so their results stay
    comparable. Patience defaults come from config.py (EARLY_STOP_PATIENCE
    / PLATEAU_PATIENCE) rather than being hardcoded here, so they always
    stay consistent with config.EPOCHS. Returns the Keras History object."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    early_stop_patience = config.EARLY_STOP_PATIENCE if early_stop_patience is None else early_stop_patience
    plateau_patience = config.PLATEAU_PATIENCE if plateau_patience is None else plateau_patience

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=early_stop_patience,
                      restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=plateau_patience,
                           min_lr=1e-6, verbose=verbose),
    ]
    return model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose,
    )


def maybe_enable_mixed_precision():
    """Enables float16 mixed-precision training if config.ENABLE_MIXED_PRECISION
    is True -- a meaningful speedup on Tensor-Core GPUs (RTX 20xx and newer,
    including the 3060/3080 Ti) since a large fraction of the matmuls in an
    LSTM/attention/transformer run in float16 instead of float32. Must be
    called BEFORE building the model (sets a global Keras policy). Safe
    no-op on CPU or pre-Tensor-Core GPUs -- just won't speed anything up
    there. Each model's final Dense layer is kept in float32 (standard
    practice) for numerical stability in the loss computation."""
    if not config.ENABLE_MIXED_PRECISION:
        return
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    logger.info("Mixed precision (float16) enabled -- final output layers stay float32.")


def run_model_stage(model_name: str, run_fn, summary=None, force: bool = False):
    """Runs one model's run_fn() (its module-level `run`) wrapped in
    try/except, so a single model failing is logged and recorded, not
    fatal to the rest of the pipeline. `summary`, if given, is a
    src.utils.logging_setup.RunSummary that main.py prints at the end.
    Returns the model's metrics dict on success, None on failure."""
    try:
        logger.info(f"=== {model_name} ===")
        metrics = run_fn(force=force)
        detail = ""
        if isinstance(metrics, dict) and "mae" in metrics:
            detail = f"MAE={metrics['mae']:.4f}"
        if summary:
            summary.add(model_name, "ok", detail)
        return metrics
    except Exception as e:
        logger.exception(f"{model_name} failed")
        if summary:
            summary.add(model_name, "failed", str(e))
        return None
