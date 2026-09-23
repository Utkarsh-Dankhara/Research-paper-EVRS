"""
src/models/bayesian_bilstm.py

The primary model -- Bayesian BiLSTM + Temporal Attention + MC Dropout.
Ported from Phase 2 and Phase 3 of notebooks/EVRS_NN_multi_models.ipynb:

  - MCDropout layer (stays active at inference -> epistemic uncertainty)
  - TemporalAttention layer (Bahdanau-style additive attention)
  - build_model(): BiLSTM(128) -> BiLSTM(64) -> attention -> Dense head
  - Training with Huber loss, EarlyStopping, ReduceLROnPlateau
  - Monte Carlo inference (config.NUM_PASSES) for epistemic uncertainty
  - Temperature scaling calibration (ECE before/after, fit on val set)
  - Phase 3: full-graph inference over every road in all 3 cities,
    producing the routing map that src/routing/ consumes later

Saves:
  models/bayesian_bilstm/model.keras
  models/bayesian_bilstm/temperature_scale.joblib
  results/bayesian_bilstm/{metrics.json, predictions.npz, history.json,
                            plots/training_curves.png, plots/eval_bilstm.png}
  data/routing/bayesian_routing_map.csv
  data/routing/highway_risk_fallback.json   (used by src/routing/router.py
                                              to weight any graph edge not
                                              covered in the routing map)
"""
import logging

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM, Layer
from tensorflow.keras.models import Model, load_model

import config
from src.data.loaders import load_sequence_cache
from src.evaluation.metrics import (
    expected_calibration_error, fit_temperature_scale, regression_metrics,
)
from src.evaluation.plot_utils import plot_neural_model_eval, plot_training_curves
from src.models.base import compile_and_train_keras, maybe_enable_mixed_precision
from src.utils.io import ensure_project_dirs, output_exists, save_json

logger = logging.getLogger(__name__)

MODEL_NAME = "bayesian_bilstm"
MODEL_PATH = config.MODELS_DIR / MODEL_NAME / "model.keras"
TEMP_SCALE_PATH = config.MODELS_DIR / MODEL_NAME / "temperature_scale.joblib"
RESULTS_DIR = config.RESULTS_DIR / MODEL_NAME


# ============================================================
# Architecture
# ============================================================

class MCDropout(Dropout):
    """Monte Carlo Dropout -- stays active at inference for epistemic uncertainty."""

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


class TemporalAttention(Layer):
    """Bahdanau-style additive attention over the predecessor sequence."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(units, use_bias=False)
        self.V = Dense(1, use_bias=False)

    def build(self, input_shape):
        self.W.build(input_shape)
        v_input_shape = tuple(input_shape[:-1]) + (self.W.units,)
        self.V.build(v_input_shape)
        super().build(input_shape)

    def call(self, hidden_states):
        score = self.V(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.W.units})
        return cfg


def _configure_gpu():
    """
    Enable memory growth before any layer is built.
    FIX: Without this TF allocates all VRAM at startup, causing OOM on
    shared machines. Must be called before any tf.keras layer instantiation.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU detected — training on CPU.")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Memory growth enabled: {gpu.name}")
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth on {gpu.name}: {e}")
    logger.info(f"{len(gpus)} GPU(s) available.")


def _residual_block(x, units: int, dropout_rate: float, block_name: str):
    """
    Residual Dense block with LayerNorm and MCDropout.

    FIX: The flat Dense(64)→Dense(32) head learned a near-linear mapping
    from the attention context to the risk score — the easiest solution
    when the label was also near-linear. Residual blocks with LayerNorm
    force nonlinear transformations: each block learns the residual
    correction to the previous representation, not a direct mapping.

    LayerNorm (not BatchNorm): MC-Dropout runs many single-sample forward
    passes. BatchNorm statistics are noisy / wrong at batch_size=1.
    LayerNorm normalises per-sample and is batch-size invariant.
    """
    from tensorflow.keras.layers import Add, LayerNormalization
    if x.shape[-1] != units:
        x = Dense(units, activation=None, name=f"{block_name}_proj")(x)
    residual = x
    x = Dense(units, activation="relu", name=f"{block_name}_d1")(x)
    x = LayerNormalization(epsilon=1e-6, name=f"{block_name}_ln")(x)
    x = MCDropout(dropout_rate, name=f"{block_name}_drop")(x)
    x = Dense(units, activation="relu", name=f"{block_name}_d2")(x)
    x = Add(name=f"{block_name}_skip")([x, residual])
    return x


def build_model(seq_len, n_features):
    """
    Bayesian BiLSTM + Temporal Attention + Residual Dense head.
    Input shape: (batch, SEQ_LEN=5, n_features=22)
    Output: sigmoid in (0,1) matching risk_score_v2 ∈ [0,1]
    """
    inp = Input(shape=(seq_len, n_features), name="road_sequence")

    x = Bidirectional(
        LSTM(config.LSTM_UNITS, return_sequences=True, activation="tanh"),
        name="bilstm_1",
    )(inp)
    x = MCDropout(config.DROPOUT_RATE, name="mc_dropout_1")(x)

    x = Bidirectional(
        LSTM(config.LSTM_UNITS // 2, return_sequences=True, activation="tanh"),
        name="bilstm_2",
    )(x)
    x = MCDropout(config.DROPOUT_RATE, name="mc_dropout_2")(x)

    context, attn_weights = TemporalAttention(config.LSTM_UNITS // 2, name="temporal_attention")(x)

    # FIX: residual blocks replace flat Dense(64)→Dense(32).
    # Flat head collapses to near-linear mapping. Residual blocks with
    # LayerNorm enforce nonlinear representations at every layer.
    x = _residual_block(context, units=128, dropout_rate=config.DROPOUT_RATE, block_name="res_1")
    x = _residual_block(x,       units=64,  dropout_rate=config.DROPOUT_RATE, block_name="res_2")

    # FIX: was activation="sigmoid" on the reasoning that risk_score_v2 in
    # [0,1] means sigmoid is a natural fit. But this model trains against
    # y_train/y_val from the SAME shared cache every other model uses --
    # RobustScaler-transformed risk_score_v2, not the raw [0,1] value.
    # RobustScaler centers on the median (~0.5) and scales by the IQR
    # (~0.5 for risk_score_v2's roughly-uniform distribution), so roughly
    # HALF the training targets land as negative numbers in the space
    # actually being fit -- values a sigmoid output can never reach.
    # linear matches what vanilla_lstm.py and transformer.py already do
    # correctly on this identical target scaling.
    out = Dense(1, activation="linear", name="risk_prediction", dtype="float32")(x)

    return Model(inputs=inp, outputs=out)


_CUSTOM_OBJECTS = {"MCDropout": MCDropout, "TemporalAttention": TemporalAttention}


def _mc_predict(attn_model, X, num_passes=config.NUM_PASSES, batch_size=config.BATCH_SIZE, label="MC inference"):
    """Runs num_passes stochastic forward passes (MC Dropout stays active)
    and returns (mean_scaled, std_scaled) across the passes.

    Calls the model directly (attn_model(batch, training=True)) instead
    of Keras's high-level .predict(). .predict() rebuilds its internal
    data-adapter/callback pipeline on every call, which is expensive to
    repeat num_passes times in a tight loop and is what caused the
    "N calls triggered tf.function retracing" warning during testing --
    retracing is very slow. Direct __call__ is the standard fix for
    exactly this "call the same model many times on the same data"
    pattern and should be substantially faster, especially on GPU.

    Logs progress every ~1/4 of the passes (not every 25 -- that never
    fired at all when num_passes=20, the default, which is why this
    phase previously produced zero visible output)."""
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    n = int(X_tensor.shape[0])
    preds = np.zeros((num_passes, n), dtype=np.float32)
    log_every = max(1, num_passes // 4)

    for i in range(num_passes):
        batch_outputs = []
        for start in range(0, n, batch_size):
            batch = X_tensor[start:start + batch_size]
            raw, _ = attn_model(batch, training=True)
            batch_outputs.append(raw.numpy()[:, 0])
        preds[i] = np.concatenate(batch_outputs)
        if (i + 1) % log_every == 0 or (i + 1) == num_passes:
            logger.info(f"  [{label}] MC pass {i + 1}/{num_passes}")

    return preds.mean(axis=0), preds.std(axis=0)


# ============================================================
# Phase 3 -- full-graph inference -> routing map
# ============================================================

def _full_graph_inference(attn_model, y_scaler, T_calib, row_meta):
    """Runs MC inference over EVERY road segment (not just the test
    split), so every edge in the routing graph gets a predicted risk +
    calibrated uncertainty. Builds the highway-type mean-risk fallback
    (for any graph edge the router encounters that isn't in this map,
    e.g. after a future filtering change) and saves the combined routing
    map used by src/routing/."""
    cache = load_sequence_cache(include_full=True)
    X_full = cache["X_full"]

    logger.info(f"Running {config.NUM_PASSES} MC passes on full dataset ({len(X_full)} segments)...")
    mean_pred_scaled, std_pred_scaled = _mc_predict(attn_model, X_full, label="full-graph inference")

    mean_pred_real = y_scaler.inverse_transform(mean_pred_scaled.reshape(-1, 1))[:, 0]
    std_pred_real = std_pred_scaled * y_scaler.scale_[0] * T_calib

    logger.info(f"Full-map inference complete: ai_predicted_risk mean={mean_pred_real.mean():.4f}, "
                f"epistemic sigma (calibrated) mean={std_pred_real.mean():.4f}")

    highway_mean_risk = (
        row_meta.assign(pred_risk=mean_pred_real)
        .groupby("highway")["pred_risk"]
        .mean()
        .to_dict()
    )
    global_mean_risk = float(mean_pred_real.mean())
    save_json(config.ROUTING_DATA_DIR / "highway_risk_fallback.json", {
        "by_highway": highway_mean_risk, "global": global_mean_risk,
    })

    routing_df = row_meta[["osmid", "osmid_local", "city_code", "city", "highway", "travel_time"]].copy()
    routing_df["ai_predicted_risk"] = mean_pred_real
    routing_df["epistemic_uncertainty"] = std_pred_real
    routing_df["uncertainty_pct_rank"] = pd.Series(std_pred_real).rank(pct=True).values
    # Bayesian edge weight: predicted_risk x travel_time (+ alpha * sigma, applied at routing time)
    routing_df["bayesian_base_cost"] = routing_df["ai_predicted_risk"] * routing_df["travel_time"]

    config.ROUTING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    routing_df.to_csv(config.ROUTING_MAP_CSV, index=False)
    logger.info(f"Saved routing map -> {config.ROUTING_MAP_CSV} ({len(routing_df):,} rows)")
    for code, n in routing_df["city_code"].value_counts().items():
        logger.info(f"    {code}: {n:,} rows")

    return routing_df


# ============================================================
# Public entry point
# ============================================================

def run(force: bool = False) -> dict:
    """
    Trains the Bayesian BiLSTM, evaluates + calibrates its uncertainty,
    and runs Phase 3 full-graph inference to produce the routing map.

    Skips training (loads the existing model + metrics.json) if
    MODEL_PATH already exists and force=False. Standalone:
        python -m src.models.bayesian_bilstm
    """
    routing_map_exists = output_exists(config.ROUTING_MAP_CSV)

    if output_exists(MODEL_PATH) and not force:
        if routing_map_exists:
            logger.info(f"Skipping training -- {MODEL_PATH} and the routing map both already "
                        f"exist (pass force=True to rerun).")
            if output_exists(RESULTS_DIR / "metrics.json"):
                from src.utils.io import load_json
                return load_json(RESULTS_DIR / "metrics.json")
            return {}

        logger.warning(f"{MODEL_PATH} exists but {config.ROUTING_MAP_CSV} is missing "
                        f"(routing/ was probably cleared separately) -- regenerating Phase 3 "
                        f"full-graph inference only, no retraining needed.")
        model = load_trained_model()
        attn_layer = model.get_layer("temporal_attention")
        attn_model = Model(inputs=model.input, outputs=[model.output, attn_layer.output[1]])
        cache = load_sequence_cache()
        y_scaler = cache["y_scaler"]
        T_calib = joblib.load(TEMP_SCALE_PATH) if output_exists(TEMP_SCALE_PATH) else 1.0
        row_meta = load_sequence_cache(include_full=True)["row_meta"]
        _full_graph_inference(attn_model, y_scaler, T_calib, row_meta)

        if output_exists(RESULTS_DIR / "metrics.json"):
            from src.utils.io import load_json
            return load_json(RESULTS_DIR / "metrics.json")
        return {}

    ensure_project_dirs()
    _configure_gpu()              # FIX: must be before any layer build
    maybe_enable_mixed_precision()
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)

    cache = load_sequence_cache()
    X_train, X_val, X_test = cache["X_train"], cache["X_val"], cache["X_test"]
    y_train, y_val, y_test = cache["y_train"], cache["y_val"], cache["y_test"]
    y_scaler = cache["y_scaler"]
    feature_cols = cache["feature_cols"]

    logger.info("Building Bayesian BiLSTM...")
    model = build_model(config.SEQ_LEN, len(feature_cols))
    model.summary(print_fn=logger.info)

    logger.info("Training Bayesian BiLSTM...")
    history = compile_and_train_keras(model, X_train, y_train, X_val, y_val)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"Model saved -> {MODEL_PATH}")

    plot_training_curves(history.history, config.MODEL_LABELS[MODEL_NAME],
                          RESULTS_DIR / "plots" / "training_curves.png")

    # ---------- MC inference + evaluation on test set ----------
    attn_layer = model.get_layer("temporal_attention")
    attn_model = Model(inputs=model.input, outputs=[model.output, attn_layer.output[1]])

    logger.info(f"Running {config.NUM_PASSES} Monte Carlo passes on test set...")
    y_pred_scaled_mean, y_pred_scaled_std = _mc_predict(attn_model, X_test, label="test-set eval")

    y_pred_real = y_scaler.inverse_transform(y_pred_scaled_mean.reshape(-1, 1))[:, 0]
    y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]
    y_pred_std_real = y_pred_scaled_std * y_scaler.scale_[0]

    point_metrics = regression_metrics(y_test_real, y_pred_real)
    logger.info(f"MODEL EVALUATION (test set): MAE={point_metrics['mae']:.4f}  "
                f"RMSE={point_metrics['rmse']:.4f}  R2={point_metrics['r2']:.4f}  "
                f"mean sigma={y_pred_std_real.mean():.4f}")

    calib_before = expected_calibration_error(y_test_real, y_pred_real, y_pred_std_real)
    logger.info(f"Calibration before temperature scaling: ECE={calib_before['ece']:.4f} "
                f"(50%CI={calib_before['coverage_50']*100:.1f}%, "
                f"80%CI={calib_before['coverage_80']*100:.1f}%, "
                f"95%CI={calib_before['coverage_95']*100:.1f}%)")

    # ---------- temperature scaling, fit on validation set ----------
    logger.info(f"Running {config.NUM_PASSES} Monte Carlo passes on validation set (for calibration)...")
    y_val_mean_scaled, y_val_std_scaled = _mc_predict(attn_model, X_val, label="val-set calibration")
    y_val_mean_real = y_scaler.inverse_transform(y_val_mean_scaled.reshape(-1, 1))[:, 0]
    y_val_std_real = y_val_std_scaled * y_scaler.scale_[0]
    y_val_real = y_scaler.inverse_transform(y_val.reshape(-1, 1))[:, 0]

    T_calib = fit_temperature_scale(y_val_real, y_val_mean_real, y_val_std_real)
    TEMP_SCALE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(T_calib, TEMP_SCALE_PATH)

    y_pred_std_calibrated = y_pred_std_real * T_calib
    calib_after = expected_calibration_error(y_test_real, y_pred_real, y_pred_std_calibrated)
    logger.info(f"Temperature scaling: T={T_calib:.4f} | ECE improved: "
                f"{calib_before['ece']:.4f} -> {calib_after['ece']:.4f}")

    # ---------- save metrics/predictions/history ----------
    metrics = {
        **point_metrics,
        "mean_sigma": float(y_pred_std_real.mean()),
        "max_sigma": float(y_pred_std_real.max()),
        "temperature_calibration": T_calib,
        "calibration_before": calib_before,
        "calibration_after": calib_after,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS_DIR / "metrics.json", metrics)
    save_json(RESULTS_DIR / "history.json", {k: [float(v) for v in vals] for k, vals in history.history.items()})
    np.savez(RESULTS_DIR / "predictions.npz",
             y_true=y_test_real, y_pred=y_pred_real,
             y_pred_std=y_pred_std_real, y_pred_std_calibrated=y_pred_std_calibrated)

    plot_neural_model_eval(y_test_real, y_pred_real, history.history,
                            config.MODEL_LABELS[MODEL_NAME], config.MODEL_COLORS[MODEL_NAME],
                            RESULTS_DIR / "plots" / "eval_bilstm.png")

    # ---------- Phase 3: full-graph inference -> routing map ----------
    row_meta = load_sequence_cache(include_full=True)["row_meta"]
    _full_graph_inference(attn_model, y_scaler, T_calib, row_meta)

    return metrics


def load_trained_model():
    """Loads the saved model with its custom layers registered, for
    inference elsewhere (e.g. the dashboard) without retraining.

    FIX: TemporalAttention builds its own Dense sublayers (self.W, self.V)
    fresh inside __init__ rather than restoring them from a saved
    sub-config. A freshly-constructed Dense layer picks up whatever the
    GLOBAL mixed-precision policy is at that exact moment, not whatever
    policy the model was originally trained under. If this model was
    trained with ENABLE_MIXED_PRECISION=True but reloaded without first
    re-enabling that global policy, TemporalAttention's Dense sublayers
    reconstruct as plain float32 while the surrounding BiLSTM output
    (whose own dtype IS preserved via its own complete get_config())
    correctly comes back as float16 -- producing a
    "float16 does not match float32" TypeError inside TemporalAttention.call()
    at deserialization time. Setting the policy before load_model() avoids
    this, matching what the training branch above already does before
    building the model.
    """
    maybe_enable_mixed_precision()
    return load_model(MODEL_PATH, custom_objects=_CUSTOM_OBJECTS)


if __name__ == "__main__":
    from src.utils.cli import parse_force_flag
    from src.utils.logging_setup import get_logger
    get_logger()
    run(force=parse_force_flag())