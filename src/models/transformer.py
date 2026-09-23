"""
src/models/transformer.py

Baseline Model 4 -- multi-head self-attention Transformer encoder with
sinusoidal positional encoding over the SEQ_LEN-step predecessor
sequence. Directly competes with the BiLSTM + Temporal Attention
architecture as the most relevant modern sequence baseline. Ported from
the "Model 4 -- Transformer (Multi-Head Self-Attention)" section of
notebooks/EVRS_NN_multi_models.ipynb.

Fully self-contained: loads the cached sequence data itself, trains,
evaluates, and writes its own results -- does not depend on any other
model having run first or successfully.

Saves:
  models/transformer/model.keras
  results/transformer/{metrics.json, predictions.npz, history.json,
                        plots/eval_transformer.png}

Standalone: python -m src.models.transformer
"""
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Add, Dense, Dropout, GlobalAveragePooling1D, Input,
    LayerNormalization, MultiHeadAttention,
)
from tensorflow.keras.models import Model, load_model

import config
from src.data.loaders import load_sequence_cache
from src.evaluation.metrics import regression_metrics
from src.evaluation.plot_utils import plot_neural_model_eval
from src.models.base import compile_and_train_keras, maybe_enable_mixed_precision
from src.utils.io import ensure_project_dirs, load_json, output_exists, save_json

logger = logging.getLogger(__name__)

MODEL_NAME = "transformer"
MODEL_PATH = config.MODELS_DIR / MODEL_NAME / "model.keras"
RESULTS_DIR = config.RESULTS_DIR / MODEL_NAME


def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding -- gives the model sequence order awareness."""
    pos = np.arange(seq_len)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return tf.cast(pe[np.newaxis, :, :], dtype=tf.float32)


def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate):
    attn_out = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))   # residual + norm
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout_rate)(ff)
    ff = Dense(d_model)(ff)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))         # residual + norm
    return x


def build_model(seq_len, n_features, d_model=64, num_heads=4,
                 ff_dim=128, num_blocks=2, dropout_rate=None):
    dropout_rate = config.DROPOUT_RATE if dropout_rate is None else dropout_rate
    inp = Input(shape=(seq_len, n_features), name="road_sequence")
    x = Dense(d_model, activation="relu", name="input_projection")(inp)
    x = x + positional_encoding(seq_len, d_model)
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="linear", name="risk_prediction", dtype="float32")(x)
    return Model(inputs=inp, outputs=out)


def run(force: bool = False) -> dict:
    """
    Trains the Transformer baseline and evaluates it on the identical
    held-out test set used by every other model.

    Skips training (loads the existing model + metrics.json) if
    MODEL_PATH already exists and force=False.
    """
    if output_exists(MODEL_PATH) and not force:
        logger.info(f"Skipping training -- {MODEL_PATH} already exists (pass force=True to rerun).")
        if output_exists(RESULTS_DIR / "metrics.json"):
            return load_json(RESULTS_DIR / "metrics.json")
        return {}

    ensure_project_dirs()
    maybe_enable_mixed_precision()
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)

    cache = load_sequence_cache()
    X_train, X_val, X_test = cache["X_train"], cache["X_val"], cache["X_test"]
    y_train, y_val, y_test = cache["y_train"], cache["y_val"], cache["y_test"]
    y_scaler = cache["y_scaler"]
    feature_cols = cache["feature_cols"]

    logger.info("Building Transformer...")
    # d_model kept small (64) to avoid overfitting on a few thousand train samples
    model = build_model(config.SEQ_LEN, len(feature_cols), d_model=64,
                         num_heads=4, ff_dim=128, num_blocks=2)
    model.summary(print_fn=logger.info)

    logger.info("Training Transformer...")
    history = compile_and_train_keras(model, X_train, y_train, X_val, y_val)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"Model saved -> {MODEL_PATH}")

    pred_scaled = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)[:, 0]
    y_pred_real = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[:, 0]
    y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]

    metrics = regression_metrics(y_test_real, y_pred_real)
    logger.info(f"Transformer: MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS_DIR / "metrics.json", metrics)
    save_json(RESULTS_DIR / "history.json", {k: [float(v) for v in vals] for k, vals in history.history.items()})
    np.savez(RESULTS_DIR / "predictions.npz", y_true=y_test_real, y_pred=y_pred_real)

    plot_neural_model_eval(
        y_test_real, y_pred_real, history.history,
        model_label=config.MODEL_LABELS[MODEL_NAME],
        color=config.MODEL_COLORS[MODEL_NAME],
        save_path=RESULTS_DIR / "plots" / "eval_transformer.png",
    )

    return metrics


def load_trained_model():
    # FIX: positional_encoding is a plain function returning a tf.Tensor,
    # NOT a Keras Layer. Passing it in custom_objects causes a TypeError
    # on load. The encoding is baked into the saved weights; no registration needed.
    #
    # FIX: also re-enable the mixed-precision policy before deserializing
    # -- see the matching note in bayesian_bilstm.py::load_trained_model()
    # for why a reload without this can crash on a custom layer with
    # freshly-constructed sublayers.
    maybe_enable_mixed_precision()
    return load_model(MODEL_PATH)


if __name__ == "__main__":
    from src.utils.cli import parse_force_flag
    from src.utils.logging_setup import get_logger
    get_logger()
    run(force=parse_force_flag())