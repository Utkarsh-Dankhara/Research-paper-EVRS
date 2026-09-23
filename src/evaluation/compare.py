"""
src/evaluation/compare.py

Builds the COMBINED comparison across whichever models actually have
saved results -- reads results/<model>/metrics.json for each of
config.MODEL_NAMES and silently skips any that are missing (a model
that failed to train, or hasn't been trained yet), instead of crashing.
This is what lets "3 of 5 models trained" still produce a complete,
useful comparison instead of an all-or-nothing failure.

Ported from the "Master Comparison Figure" sections of
notebooks/EVRS_NN_multi_models.ipynb (cells 41/59/60/61):
  - Panel A: grouped bar (MAE / RMSE / R2)
  - Panel B: predicted-vs-actual scatter, all models overlaid
  - Panel C: residual distribution KDEs
  - Panel D: validation-loss convergence (neural models only)
  - metrics_summary.csv, incl. Bayesian-vs-each-baseline %% MAE improvement
  - 2-row supplementary grid: one column per available model

Saves to results/comparison/:
  model_comparison_master.png
  model_comparison_grid.png
  metrics_summary.csv
  comparison_summary.json

Runnable standalone: python -m src.evaluation.compare
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import config
from src.utils.io import ensure_project_dirs, load_json, save_json

logger = logging.getLogger(__name__)


# ============================================================
# Loading whatever results actually exist
# ============================================================

def _load_model_results(model_name: str):
    """Loads one model's metrics/predictions/history from results/<model>/
    if they exist. Returns None if that model hasn't been trained (or
    failed), OR if its saved files are corrupted/incompatible -- this is
    the piece that lets the comparison degrade gracefully instead of
    requiring all 5 models to have clean results."""
    results_dir = config.RESULTS_DIR / model_name
    metrics_path = results_dir / "metrics.json"
    preds_path = results_dir / "predictions.npz"

    if not (metrics_path.exists() and preds_path.exists()):
        return None

    try:
        metrics = load_json(metrics_path)
        preds = np.load(preds_path)

        history = None
        history_path = results_dir / "history.json"
        if history_path.exists():
            history = load_json(history_path)

        return {
            "label": config.MODEL_LABELS[model_name],
            "color": config.MODEL_COLORS[model_name],
            "mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"],
            "y_true": preds["y_true"], "y_pred": preds["y_pred"],
            "history": history,
        }
    except Exception as e:
        logger.warning(f"'{model_name}' has results on disk but they're corrupted or "
                        f"incompatible ({e}) -- excluding it from the comparison rather than "
                        f"failing the whole comparison. Retrain it to fix this.")
        return None


def _collect_available_results() -> dict:
    """Returns {model_name: result_dict} for every model that actually
    has saved results, logging (not raising) for any that are missing."""
    store = {}
    for name in config.MODEL_NAMES:
        r = _load_model_results(name)
        if r is None:
            logger.warning(f"No results found for '{name}' -- skipping it in the comparison "
                            f"(train it with `python -m src.models.{name}` to include it).")
            continue
        store[name] = r
    return store


# ============================================================
# Panel builders (each works with however many models are available)
# ============================================================

def _panel_bar_metrics(ax, store, order):
    x = np.arange(len(order))
    w = 0.26
    maes = [store[k]["mae"] for k in order]
    rmses = [store[k]["rmse"] for k in order]
    r2s = [store[k]["r2"] for k in order]

    bar_groups = [
        ax.bar(x - w, maes, w, label="MAE (lower better)", color="#5C6BC0", alpha=0.88),
        ax.bar(x, rmses, w, label="RMSE (lower better)", color="#EF5350", alpha=0.88),
        ax.bar(x + w, r2s, w, label="R2 (higher better)", color="#26A69A", alpha=0.88),
    ]
    for group in bar_groups:
        for bar in group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    if "bayesian_bilstm" in order:
        bx = x[order.index("bayesian_bilstm")]
        ax.axvspan(bx - w * 1.85, bx + w * 1.85, alpha=0.09,
                   color=config.MODEL_COLORS["bayesian_bilstm"], zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([store[k]["label"] for k in order], fontsize=9)
    ax.set_ylabel("Metric value")
    ax.set_title("(A)  MAE . RMSE . R2", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(maes), max(rmses), max(r2s)) * 1.18)


def _panel_scatter(ax, store, order):
    all_true = np.concatenate([store[k]["y_true"] for k in order])
    lo, hi = float(all_true.min()), float(all_true.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, zorder=5, label="Perfect")

    for k in order:
        r = store[k]
        is_primary = k == "bayesian_bilstm"
        ax.scatter(r["y_true"], r["y_pred"], color=r["color"],
                   s=28 if is_primary else 18, alpha=0.85 if is_primary else 0.55,
                   zorder=10 if is_primary else 3, label=r["label"])

    ax.set_xlabel("Actual risk score")
    ax.set_ylabel("Predicted risk score")
    ax.set_title("(B)  Predicted vs Actual", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, markerscale=1.4)
    ax.grid(alpha=0.3)


def _panel_residual_kde(ax, store, order):
    for k in order:
        r = store[k]
        res = r["y_pred"] - r["y_true"]
        if len(res) < 2 or np.isclose(res.std(), 0):
            logger.warning(f"'{k}' residuals have ~zero variance -- skipping its KDE curve.")
            continue
        xk = np.linspace(res.min() - 0.15, res.max() + 0.15, 300)
        kde = gaussian_kde(res, bw_method="scott")
        is_primary = k == "bayesian_bilstm"
        ax.plot(xk, kde(xk), color=r["color"],
                lw=2.8 if is_primary else 1.6, ls="-" if is_primary else "--",
                label=r["label"])
        ax.axvline(res.mean(), color=r["color"], lw=0.75, alpha=0.45)

    ax.axvline(0, color="k", lw=1.4, ls=":", label="Zero error")
    ax.set_xlabel("Residual (pred - actual)")
    ax.set_ylabel("Density")
    ax.set_title("(C)  Residual Distributions (KDE)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _panel_val_loss(ax, store, order):
    neural = [k for k in order if store[k]["history"] is not None]
    if not neural:
        ax.text(0.5, 0.5, "No neural model training history available", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("(D)  Validation Loss Convergence", fontsize=11, fontweight="bold")
        return

    for k in neural:
        r = store[k]
        vl = r["history"]["val_loss"]
        is_primary = k == "bayesian_bilstm"
        ax.plot(range(1, len(vl) + 1), vl, color=r["color"],
                lw=2.6 if is_primary else 1.8, ls="-" if is_primary else "--",
                label=r["label"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Huber Loss")
    ax.set_title("(D)  Validation Loss Convergence", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def _build_master_figure(store, order, save_path):
    """2x2 master comparison figure. Matches the original notebook's
    model_comparison_master.png, but built from however many models are
    actually available instead of assuming all 5."""
    fig = plt.figure(figsize=(20, 13))
    n_missing = len(config.MODEL_NAMES) - len(order)
    title = "Model Comparison - Risk Score Prediction Across Gandhinagar, Ahmedabad, and Surat"
    if n_missing:
        title += f"  ({len(order)}/{len(config.MODEL_NAMES)} models -- {n_missing} unavailable)"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.32,
                            top=0.92, bottom=0.08, left=0.07, right=0.97)
    _panel_bar_metrics(fig.add_subplot(gs[0, 0]), store, order)
    _panel_scatter(fig.add_subplot(gs[0, 1]), store, order)
    _panel_residual_kde(fig.add_subplot(gs[1, 0]), store, order)
    _panel_val_loss(fig.add_subplot(gs[1, 1]), store, order)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _build_supplementary_grid(store, order, save_path):
    """Per-model 2-row grid: predicted-vs-actual (top) + residual
    histogram (bottom), one column per AVAILABLE model -- width adapts
    instead of assuming a fixed 5 columns."""
    n = len(order)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 8), squeeze=False)
    fig.suptitle("Per-Model: Predicted vs Actual (top row) . Residual Distribution (bottom row)",
                 fontsize=13, fontweight="bold")

    for col, k in enumerate(order):
        r = store[k]
        color = r["color"]
        res = r["y_pred"] - r["y_true"]
        is_primary = k == "bayesian_bilstm"

        ax_s = axes[0, col]
        ax_s.scatter(r["y_true"], r["y_pred"], alpha=0.35, s=14, color=color)
        lo, hi = float(r["y_true"].min()), float(r["y_true"].max())
        ax_s.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax_s.set_title(f"{r['label']}\nMAE={r['mae']:.4f}   R2={r['r2']:.4f}",
                       fontsize=9, fontweight="bold" if is_primary else "normal")
        ax_s.set_xlabel("Actual")
        ax_s.set_ylabel("Predicted" if col == 0 else "")
        ax_s.grid(alpha=0.3)
        if is_primary:
            for sp in ax_s.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(2.5)

        ax_r = axes[1, col]
        ax_r.hist(res, bins=35, color=color, edgecolor="white", alpha=0.85)
        ax_r.axvline(0, color="k", lw=1.2, ls="--")
        ax_r.axvline(res.mean(), color="red", lw=1.2, ls=":", label=f"mean={res.mean():.3f}")
        ax_r.set_xlabel("Residual")
        ax_r.set_ylabel("Count" if col == 0 else "")
        ax_r.legend(fontsize=7)
        ax_r.grid(alpha=0.3)
        if is_primary:
            for sp in ax_r.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(2.5)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _build_summary_table(store, order) -> pd.DataFrame:
    """Metrics summary table -- one row per available model, including
    the sequence-context / uncertainty-estimate qualitative columns from
    the original notebook, plus a Bayesian-vs-baseline %% MAE improvement
    column whenever the Bayesian model is among the available results."""
    bilstm = store.get("bayesian_bilstm")
    rows = []
    for k in order:
        v = store[k]
        row = {
            "model": v["label"],
            "mae": round(v["mae"], 4),
            "rmse": round(v["rmse"], 4),
            "r2": round(v["r2"], 4),
            "sequence_context": k in ("vanilla_lstm", "transformer", "bayesian_bilstm"),
            "uncertainty_estimate": k == "bayesian_bilstm",
        }
        if bilstm and k != "bayesian_bilstm":
            row["bayesian_mae_improvement_pct"] = round(100 * (v["mae"] - bilstm["mae"]) / v["mae"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Public entry point
# ============================================================

def run() -> dict:
    """
    Builds the combined comparison across every model that has results
    on disk. Never raises just because a model is missing -- if only 1
    of the 5 models trained successfully, this still produces a (smaller)
    comparison instead of failing.

    Returns {"available_models": [...], "missing_models": [...], ...}.
    Standalone: python -m src.evaluation.compare
    """
    ensure_project_dirs()

    store = _collect_available_results()
    if not store:
        logger.error("No model results found at all -- train at least one model first, "
                      "e.g. `python -m src.models.random_forest`. Comparison skipped.")
        result = {"available_models": [], "missing_models": config.MODEL_NAMES}
        save_json(config.COMPARISON_DIR / "comparison_summary.json", result)
        return result

    order = [m for m in config.COMPARISON_ORDER if m in store]
    missing = [m for m in config.MODEL_NAMES if m not in store]
    logger.info(f"Building comparison from {len(order)}/{len(config.MODEL_NAMES)} available models: {order}")
    if missing:
        logger.info(f"Not included (no results found): {missing}")

    _build_master_figure(store, order, config.COMPARISON_DIR / "model_comparison_master.png")
    _build_supplementary_grid(store, order, config.COMPARISON_DIR / "model_comparison_grid.png")

    summary_df = _build_summary_table(store, order)
    config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(config.COMPARISON_DIR / "metrics_summary.csv", index=False)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    best_mae = min(store, key=lambda k: store[k]["mae"])
    best_rmse = min(store, key=lambda k: store[k]["rmse"])
    best_r2 = max(store, key=lambda k: store[k]["r2"])
    logger.info(f"Best MAE  -> {store[best_mae]['label']} ({store[best_mae]['mae']:.4f})")
    logger.info(f"Best RMSE -> {store[best_rmse]['label']} ({store[best_rmse]['rmse']:.4f})")
    logger.info(f"Best R2   -> {store[best_r2]['label']} ({store[best_r2]['r2']:.4f})")

    result = {
        "available_models": order, "missing_models": missing,
        "best_mae_model": best_mae, "best_rmse_model": best_rmse, "best_r2_model": best_r2,
    }
    save_json(config.COMPARISON_DIR / "comparison_summary.json", result)
    return result


if __name__ == "__main__":
    from src.utils.logging_setup import get_logger
    get_logger()
    run()
