"""
src/evaluation/plot_utils.py

Shared per-model evaluation plot helpers, so the 5 model modules don't
each reimplement the same matplotlib code. Panel layouts match the
original notebook's per-model figures exactly (eval_rf.png, eval_xgb.png,
eval_lstm.png, eval_transformer.png, training_curves.png) so results are
visually comparable across models.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe -- these run as scripts, not in a notebook display
import matplotlib.pyplot as plt
import numpy as np


def _scatter_pred_vs_actual(ax, y_true, y_pred, color):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color=color)
    lo, hi = float(y_true.min()), float(y_true.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect")
    ax.set_xlabel("Actual risk score")
    ax.set_ylabel("Predicted risk score")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _residual_histogram(ax, y_true, y_pred, color):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    ax.hist(residuals, bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", lw=1.2, ls="--")
    ax.axvline(residuals.mean(), color="red", lw=1.2, ls=":", label=f"Mean={residuals.mean():.3f}")
    ax.set_xlabel("Residual (pred - actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _training_curve(ax, train_vals, val_vals, ylabel, title, color):
    ax.plot(train_vals, color=color, lw=2, label="Train")
    ax.plot(val_vals, color="#333333", lw=2, ls="--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _save(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scatter_residual_only(y_true, y_pred, model_label, color, save_path):
    """2-panel fallback (scatter + residual histogram only) for any model
    that doesn't expose feature importances or training history."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{model_label} - Evaluation (risk_score_v1)", fontsize=13, fontweight="bold")
    _scatter_pred_vs_actual(axes[0], y_true, y_pred, color)
    _residual_histogram(axes[1], y_true, y_pred, color)
    return _save(fig, save_path)


def plot_tree_model_eval(y_true, y_pred, importances, feature_names, model_label, color, save_path, top_n=15):
    """3-panel evaluation figure for tree models (Random Forest / XGBoost):
    scatter, residual histogram, top-N feature importance."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"{model_label} - Evaluation (risk_score_v1)", fontsize=13, fontweight="bold")

    _scatter_pred_vs_actual(axes[0], y_true, y_pred, color)
    _residual_histogram(axes[1], y_true, y_pred, color)

    ax = axes[2]
    importances = np.asarray(importances)
    top_idx = np.argsort(importances)[-top_n:]
    ax.barh(range(len(top_idx)), importances[top_idx], color=color, alpha=0.85)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=7)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top-{top_n} Feature Importances")
    ax.grid(axis="x", alpha=0.3)

    return _save(fig, save_path)


def plot_neural_model_eval(y_true, y_pred, history, model_label, color, save_path):
    """4-panel evaluation figure for neural models (Bayesian BiLSTM,
    vanilla LSTM, Transformer): loss convergence, MAE convergence,
    scatter, residual histogram. `history` is a Keras History.history
    dict (with 'loss'/'val_loss'/'mae'/'val_mae') from the just-completed
    training run."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"{model_label} - Evaluation (risk_score_v1)", fontsize=13, fontweight="bold")

    _training_curve(axes[0], history["loss"], history["val_loss"], "Huber Loss", "Training Convergence", color)
    _training_curve(axes[1], history["mae"], history["val_mae"], "MAE (scaled)", "MAE Convergence", color)
    _scatter_pred_vs_actual(axes[2], y_true, y_pred, color)
    _residual_histogram(axes[3], y_true, y_pred, color)

    return _save(fig, save_path)


def plot_training_curves(history, model_label, save_path):
    """Standalone 2-panel training-curve figure (loss + MAE). Matches the
    original notebook's training_curves.png for the Bayesian BiLSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Training convergence - {model_label} (target: risk_score_v1)", fontsize=12)

    _training_curve(axes[0], history["loss"], history["val_loss"], "Loss", "Huber Loss", "#E53935")
    _training_curve(axes[1], history["mae"], history["val_mae"], "MAE (scaled)", "Mean Absolute Error", "#E53935")

    return _save(fig, save_path)
