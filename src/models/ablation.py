"""
src/models/ablation.py

Ablation study for the Bayesian BiLSTM. Your existing 5-model comparison
(RF/XGB/vanilla LSTM/Transformer/Bayesian BiLSTM) compares different
MODEL FAMILIES -- this module instead isolates the architectural
decisions stacked inside the Bayesian BiLSTM itself, holding everything
else (features, data split, training setup) fixed.

## Mapping to the review comment

The comment asks for an ablation of: Bidirectional LSTM, Temperature
Scaling, Monte Carlo Dropout, Temporal Attention, Bayesian Routing Cost
Function. Only 3 of these are model-architecture toggles that make sense
to retrain-and-compare on point-prediction metrics -- the other 2 are
answered differently, on purpose, not by omission:

  - **Bidirectional / Attention / MC Dropout** -- `--mode reviewer` (5
    variants: full model, each removed individually, and all 3 removed
    together). Compared on MAE/RMSE/R2 exactly like the other models.
  - **Temperature Scaling** -- NOT a separate trained variant. It's a
    post-hoc recalibration of an ALREADY-trained model's uncertainty
    (fit on the val set, applied to the test set), so every mc_dropout=
    True variant automatically reports calibration BOTH before and
    after in its metrics.json (`temperature_scaling: {ece_before,
    ece_after, coverage_95_before, coverage_95_after}`). See also
    `src.evaluation.uncertainty_analysis` for the reliability-diagram
    figure this produces.
  - **Bayesian Routing Cost Function** -- NOT something a model-training
    ablation can answer at all (it's a routing-time decision, not a
    trained parameter). Already answered by the existing routing
    pipeline: `data/routing/bootstrap_summary.json` and
    `data/routing/route_comparison_metrics.csv` (from
    `src.routing.bootstrap`) directly quantify baseline-cost vs.
    Bayesian-cost routing outcomes across many OD pairs, with a
    bootstrap significance test.

## Other toggles (available, not part of the reviewer's specific ask)

  - stacked        -- 2 LSTM layers vs 1
  - seq_context     -- SEQ_LEN=5 predecessor-chain context vs SEQ_LEN=1

## Study designs

  - "reviewer" (5 variants): exactly the mapping above.
  - "leave_one_out" (7 variants): all 5 toggles (bidirectional, stacked,
    attention, seq_context, mc_dropout) individually removed, plus a
    fully-minimal baseline. Broader than the review comment asks for --
    useful for your own deeper exploration.
  - "incremental" (6 variants): minimal baseline building up to the full
    model one component at a time.
  - "full_factorial" (32 variants): every combination of the 5 toggles,
    generated programmatically. Most rigorous, most expensive.

Each variant is trained and evaluated exactly like bayesian_bilstm.py
(same Huber/EarlyStopping/ReduceLROnPlateau setup, same held-out test
set, same metrics), reusing the identical cached sequence data --
mc_dropout=False variants skip MC inference/calibration since a plain
Dropout layer has nothing to be uncertain about at inference time.

Saves:
  models/ablation/<variant_name>/model.keras
  results/ablation/<variant_name>/{metrics.json, predictions.npz, history.json}
  results/ablation/comparison/{ablation_<mode>.png, ablation_<mode>_summary.csv}

NOT wired into `main.py --stage all` -- this is a research analysis you
run deliberately, not part of the routine pipeline. Run standalone:
    python -m src.models.ablation --mode reviewer          (recommended for the review response)
    python -m src.models.ablation --mode leave_one_out
    python -m src.models.ablation --mode incremental
    python -m src.models.ablation --mode full_factorial    (32 variants -- slow)
"""
import itertools
import logging
from dataclasses import dataclass, replace

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, GlobalAveragePooling1D, Input, LSTM
from tensorflow.keras.models import Model

import config
from src.data.loaders import load_sequence_cache
from src.evaluation.metrics import expected_calibration_error, fit_temperature_scale, regression_metrics
from src.evaluation.plot_utils import _save
from src.models.base import compile_and_train_keras, maybe_enable_mixed_precision
from src.models.bayesian_bilstm import MCDropout, TemporalAttention
from src.utils.io import ensure_project_dirs, load_json, output_exists, save_json

logger = logging.getLogger(__name__)


# ============================================================
# Variant definition
# ============================================================

@dataclass(frozen=True)
class AblationVariant:
    name: str
    bidirectional: bool = True
    stacked: bool = True
    attention: bool = True
    seq_context: bool = True    # True -> full config.SEQ_LEN history, False -> current road only (length 1)
    mc_dropout: bool = True

    @property
    def seq_len(self) -> int:
        return config.SEQ_LEN if self.seq_context else 1


FULL = AblationVariant("full_model")   # identical architecture to src.models.bayesian_bilstm
MINIMAL = AblationVariant("minimal_baseline", bidirectional=False, stacked=False,
                           attention=False, seq_context=False, mc_dropout=False)

# Exactly the 3 model-architecture components named in the review comment
# "ablation evaluating: Bidirectional LSTM, Temperature Scaling, Monte
# Carlo Dropout, Temporal Attention, Bayesian Routing Cost Function".
# stacked/seq_context are held fixed at the production default (True) --
# they weren't asked about, so varying them here would just add noise to
# a specific, named request. Temperature Scaling and Bayesian Routing
# Cost Function are NOT separate trained variants (see module docstring
# for why) -- they're answered by run_variant()'s automatic before/after
# calibration reporting and by the existing routing/bootstrap pipeline
# respectively.
REVIEWER_VARIANTS = [
    FULL,
    replace(FULL, name="minus_bidirectional", bidirectional=False),
    replace(FULL, name="minus_attention", attention=False),
    replace(FULL, name="minus_mc_dropout", mc_dropout=False),
    replace(FULL, name="minus_bidirectional_attention_mc_dropout",
            bidirectional=False, attention=False, mc_dropout=False),
]

LEAVE_ONE_OUT_VARIANTS = [
    FULL,
    replace(FULL, name="minus_bidirectional", bidirectional=False),
    replace(FULL, name="minus_stacked", stacked=False),
    replace(FULL, name="minus_attention", attention=False),
    replace(FULL, name="minus_seq_context", seq_context=False),
    replace(FULL, name="minus_mc_dropout", mc_dropout=False),
    MINIMAL,
]

# Builds up from MINIMAL to FULL one component at a time -- order chosen
# to tell a natural "story": context first (data-level), then capacity
# (depth/direction), then the two smarter mechanisms (attention, then
# the Bayesian uncertainty layer) last.
INCREMENTAL_VARIANTS = [
    MINIMAL,
    replace(MINIMAL, name="plus_seq_context", seq_context=True),
    replace(MINIMAL, name="plus_seq_context_stacked", seq_context=True, stacked=True),
    replace(MINIMAL, name="plus_seq_context_stacked_bidirectional",
            seq_context=True, stacked=True, bidirectional=True),
    replace(MINIMAL, name="plus_all_but_mc_dropout",
            seq_context=True, stacked=True, bidirectional=True, attention=True),
    FULL,
]


def _full_factorial_variants():
    """All 2^5 = 32 combinations, generated programmatically rather than
    hand-listed (avoids transcription mistakes and stays correct if a
    6th toggle is ever added)."""
    toggles = ["bidirectional", "stacked", "attention", "seq_context", "mc_dropout"]
    variants = []
    for combo in itertools.product([False, True], repeat=len(toggles)):
        kwargs = dict(zip(toggles, combo))
        name = "factorial_" + "".join("1" if v else "0" for v in combo)
        variants.append(AblationVariant(name, **kwargs))
    return variants


VARIANT_SETS = {
    "reviewer": REVIEWER_VARIANTS,
    "leave_one_out": LEAVE_ONE_OUT_VARIANTS,
    "incremental": INCREMENTAL_VARIANTS,
    "full_factorial": _full_factorial_variants,   # callable -- built lazily, see run()
}


# ============================================================
# Parameterized architecture
# ============================================================

def build_variant_model(variant: AblationVariant, n_features: int):
    """Same building blocks as src.models.bayesian_bilstm.build_model, but
    with each of the 5 decisions conditionally applied based on `variant`.
    Layer widths are read from the tensor shape at build time (x.shape[-1])
    rather than precomputed, so bidirectional/stacked combinations never
    need separately-tracked width arithmetic."""
    DropoutLayer = MCDropout if variant.mc_dropout else tf.keras.layers.Dropout

    inp = Input(shape=(variant.seq_len, n_features), name="road_sequence")

    lstm1 = LSTM(config.LSTM_UNITS, return_sequences=True, activation="tanh", name="lstm_1")
    x = Bidirectional(lstm1, name="bilstm_1")(inp) if variant.bidirectional else lstm1(inp)
    x = DropoutLayer(config.DROPOUT_RATE, name="drop_1")(x)

    if variant.stacked:
        lstm2 = LSTM(config.LSTM_UNITS // 2, return_sequences=True, activation="tanh", name="lstm_2")
        x = Bidirectional(lstm2, name="bilstm_2")(x) if variant.bidirectional else lstm2(x)
        x = DropoutLayer(config.DROPOUT_RATE, name="drop_2")(x)

    if variant.attention:
        # units is TemporalAttention's own internal projection width, not
        # the input feature width -- Keras infers the actual input dim
        # automatically. Fixed at LSTM_UNITS//2 to match the reference
        # architecture in src.models.bayesian_bilstm exactly, regardless
        # of what bidirectional/stacked happen to be for this variant.
        context, _ = TemporalAttention(config.LSTM_UNITS // 2, name="temporal_attention")(x)
    else:
        context = GlobalAveragePooling1D(name="mean_pool")(x)

    d = Dense(64, activation="relu")(context)
    d = DropoutLayer(config.DROPOUT_RATE, name="drop_3")(d)
    d = Dense(32, activation="relu")(d)
    out = Dense(1, activation="linear", name="risk_prediction", dtype="float32")(d)

    return Model(inputs=inp, outputs=out, name=variant.name)


def _slice_context(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Takes the LAST seq_len steps of the cached SEQ_LEN=5 sequences.
    The predecessor chain is built target-edge-last (see
    src.data.sequence_builder._build_sequences), so X[:, -1:, :] is
    exactly 'the current road's own features, no upstream history' --
    this is what makes the seq_context ablation possible WITHOUT
    rebuilding the sequence cache with a different SEQ_LEN."""
    return X[:, -seq_len:, :]


def _mc_predict_ablation(model, X, num_passes=config.NUM_PASSES, batch_size=config.BATCH_SIZE, label=""):
    """Same MC-Dropout inference approach as
    src.models.bayesian_bilstm._mc_predict, but for a plain
    single-output model -- build_variant_model doesn't build the special
    2-output (prediction, attention_weights) wrapper bayesian_bilstm.py
    uses, since attention=False variants have no attention weights to
    expose in the first place."""
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    n = int(X_tensor.shape[0])
    preds = np.zeros((num_passes, n), dtype=np.float32)
    log_every = max(1, num_passes // 4)

    for i in range(num_passes):
        batch_outputs = []
        for start in range(0, n, batch_size):
            batch = X_tensor[start:start + batch_size]
            raw = model(batch, training=True)
            batch_outputs.append(raw.numpy()[:, 0])
        preds[i] = np.concatenate(batch_outputs)
        if (i + 1) % log_every == 0 or (i + 1) == num_passes:
            logger.info(f"  [{label}] MC pass {i + 1}/{num_passes}")

    return preds.mean(axis=0), preds.std(axis=0)


# ============================================================
# Train + evaluate one variant
# ============================================================

def run_variant(variant: AblationVariant, force: bool = False) -> dict:
    """Trains and evaluates ONE ablation variant, self-contained like
    every other model module -- reuses the same cached sequence data
    and the same training/eval setup as bayesian_bilstm.py, so results
    are directly comparable. Skips training if already done and
    force=False."""
    model_path = config.MODELS_ABLATION_DIR / variant.name / "model.keras"
    results_dir = config.RESULTS_ABLATION_DIR / variant.name

    if output_exists(model_path) and not force:
        logger.info(f"[{variant.name}] skipping -- already trained (pass force=True to rerun).")
        if output_exists(results_dir / "metrics.json"):
            return load_json(results_dir / "metrics.json")
        return {}

    maybe_enable_mixed_precision()
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)

    cache = load_sequence_cache()
    X_train = _slice_context(cache["X_train"], variant.seq_len)
    X_val = _slice_context(cache["X_val"], variant.seq_len)
    X_test = _slice_context(cache["X_test"], variant.seq_len)
    y_train, y_val, y_test = cache["y_train"], cache["y_val"], cache["y_test"]
    y_scaler = cache["y_scaler"]
    feature_cols = cache["feature_cols"]

    logger.info(f"[{variant.name}] {variant} ")
    model = build_variant_model(variant, len(feature_cols))
    history = compile_and_train_keras(model, X_train, y_train, X_val, y_val, verbose=0)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    if variant.mc_dropout:
        y_pred_scaled, y_pred_std_scaled = _mc_predict_ablation(model, X_test, label=variant.name)
        y_pred_std_real = y_pred_std_scaled * y_scaler.scale_[0]
    else:
        y_pred_scaled = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)[:, 0]
        y_pred_std_real = None

    y_pred_real = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[:, 0]
    y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]

    metrics = {
        **regression_metrics(y_test_real, y_pred_real),
        "n_params": int(model.count_params()),
        "bidirectional": variant.bidirectional, "stacked": variant.stacked,
        "attention": variant.attention, "seq_context": variant.seq_context,
        "mc_dropout": variant.mc_dropout, "seq_len": variant.seq_len,
    }
    if y_pred_std_real is not None:
        metrics["mean_sigma"] = float(y_pred_std_real.mean())

        # "Temperature Scaling" ablation -- fit on val set (same procedure as
        # bayesian_bilstm.py), then report calibration BOTH before and after.
        # No separate model needed: this is a post-hoc recalibration of the
        # SAME trained model's uncertainty, not an architecture change.
        calib_before = expected_calibration_error(y_test_real, y_pred_real, y_pred_std_real)
        y_val_scaled, y_val_std_scaled = _mc_predict_ablation(model, X_val, label=f"{variant.name}-val")
        y_val_std_real = y_val_std_scaled * y_scaler.scale_[0]
        y_val_real = y_scaler.inverse_transform(y_val.reshape(-1, 1))[:, 0]
        y_val_pred_real = y_scaler.inverse_transform(y_val_scaled.reshape(-1, 1))[:, 0]
        T_calib = fit_temperature_scale(y_val_real, y_val_pred_real, y_val_std_real)
        calib_after = expected_calibration_error(y_test_real, y_pred_real, y_pred_std_real * T_calib)

        metrics["ece"] = calib_before["ece"]                 # kept for backward-compat with existing readers
        metrics["temperature_scaling"] = {
            "temperature": T_calib,
            "ece_before": calib_before["ece"], "ece_after": calib_after["ece"],
            "coverage_95_before": calib_before["coverage_95"], "coverage_95_after": calib_after["coverage_95"],
        }

    logger.info(f"[{variant.name}] MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
                f"R2={metrics['r2']:.4f}  params={metrics['n_params']:,}")

    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / "metrics.json", metrics)
    save_json(results_dir / "history.json", {k: [float(v) for v in vals] for k, vals in history.history.items()})
    np.savez(results_dir / "predictions.npz", y_true=y_test_real, y_pred=y_pred_real)

    return metrics


# ============================================================
# Combined ablation comparison
# ============================================================

def _build_ablation_comparison(variants: list, results: dict, mode: str):
    """Bar chart of MAE across variants (in the order they were run, so
    'incremental' mode reads left-to-right as a build-up story and
    'leave_one_out' reads as deviations from the full model), plus a
    CSV summary with the delta vs the full model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    names = [v.name for v in variants if v.name in results]
    maes = [results[n]["mae"] for n in names]
    full_mae = results.get("full_model", {}).get("mae")

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5))
    colors = ["#E53935" if n in ("full_model", "minimal_baseline") else "#5C6BC0" for n in names]
    bars = ax.bar(range(len(names)), maes, color=colors, alpha=0.88)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{mae:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    if full_mae is not None:
        ax.axhline(full_mae, color="#E53935", lw=1.2, ls="--", alpha=0.6, label="full model MAE")
        ax.legend(fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title(f"Ablation study -- {mode}")
    ax.grid(axis="y", alpha=0.3)

    save_path = config.ABLATION_COMPARISON_DIR / f"ablation_{mode}.png"
    _save(fig, save_path)

    rows = []
    for n in names:
        r = results[n]
        row = {"variant": n, **{k: r[k] for k in ("mae", "rmse", "r2", "n_params") if k in r}}
        if full_mae is not None and n != "full_model":
            row["mae_delta_vs_full_pct"] = round(100 * (r["mae"] - full_mae) / full_mae, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = config.ABLATION_COMPARISON_DIR / f"ablation_{mode}_summary.csv"
    config.ABLATION_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"\n{df.to_string(index=False)}")
    return save_path, csv_path


# ============================================================
# Public entry point
# ============================================================

def run(mode: str = "reviewer", force: bool = False) -> dict:
    """Trains + evaluates every variant in the chosen study design, then
    builds the combined comparison. Each variant is wrapped in
    try/except -- one failing doesn't stop the rest (same fault-
    isolation philosophy as the main pipeline)."""
    ensure_project_dirs()

    if mode not in VARIANT_SETS:
        raise ValueError(f"Unknown mode '{mode}'. Choices: {list(VARIANT_SETS)}")
    variants = VARIANT_SETS[mode]
    if callable(variants):
        variants = variants()

    logger.info(f"Running '{mode}' ablation study: {len(variants)} variant(s)")

    results = {}
    for i, variant in enumerate(variants):
        logger.info(f"--- variant {i + 1}/{len(variants)}: {variant.name} ---")
        try:
            results[variant.name] = run_variant(variant, force=force)
        except Exception:
            logger.exception(f"[{variant.name}] failed -- continuing with the rest")

    if not results:
        logger.error("No variants completed successfully -- nothing to compare.")
        return {}

    _build_ablation_comparison(variants, results, mode)
    return results


if __name__ == "__main__":
    import argparse

    from src.utils.logging_setup import get_logger

    get_logger()
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="reviewer", choices=list(VARIANT_SETS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(mode=args.mode, force=args.force)
