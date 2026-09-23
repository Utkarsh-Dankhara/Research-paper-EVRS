"""
src/evaluation/uncertainty_analysis.py

Concrete, quantified evidence for two review comments that "the Bayesian
model is preferred despite lower point accuracy -- quantify the
operational benefit" and "calibration improvement is modest -- show
reliability diagrams/calibration plots." Both are built purely from
artifacts src.models.bayesian_bilstm.py already saves -- no retraining,
no new data.

  - plot_reliability_diagram(): nominal vs. observed coverage at
    50/80/95% CI, before AND after temperature scaling, against the
    y=x "perfectly calibrated" reference line. The standard reliability
    diagram used in uncertainty-quantification papers (e.g. Guo et al.
    2017) -- this is what "additional calibration plots" usually means.

  - plot_uncertainty_vs_error(): bins the test set into uncertainty
    tertiles (by the model's own predicted sigma) and reports actual
    MAE per bin, plus the Spearman correlation between |residual| and
    predicted sigma. This is the quantified answer to "why prefer a
    model with worse point accuracy" -- if a model's uncertainty
    estimate is meaningful, its predictions should be measurably WORSE
    exactly where it says it's unsure. A point-estimate model (Random
    Forest / XGBoost / Gradient Boosting) has no equivalent signal at
    all -- it cannot tell you which of its predictions to distrust.

Reads results/bayesian_bilstm/{metrics.json, predictions.npz}.
Saves results/uncertainty_analysis/{reliability_diagram.png,
uncertainty_vs_error.png, summary.json}.

Standalone: python -m src.evaluation.uncertainty_analysis
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.utils.io import load_json, save_json

logger = logging.getLogger(__name__)

RESULTS_DIR = config.RESULTS_DIR / "bayesian_bilstm"
OUT_DIR = config.RESULTS_DIR / "uncertainty_analysis"


def _load_bilstm_results():
    metrics_path = RESULTS_DIR / "metrics.json"
    preds_path = RESULTS_DIR / "predictions.npz"
    if not (metrics_path.exists() and preds_path.exists()):
        raise FileNotFoundError(
            f"{metrics_path} / {preds_path} not found. Train the Bayesian BiLSTM first: "
            f"`python -m src.models.bayesian_bilstm`."
        )
    metrics = load_json(metrics_path)
    preds = np.load(preds_path)
    return metrics, preds


def plot_reliability_diagram(metrics: dict, save_path):
    """Nominal vs. observed coverage at 50/80/95% CI, before and after
    temperature scaling, against the y=x perfect-calibration reference.
    A curve BELOW the diagonal means the model's intervals are too
    narrow (overconfident); ABOVE means too wide (underconfident)."""
    nominal = [0.50, 0.80, 0.95]
    before = [metrics["calibration_before"][f"coverage_{int(p*100)}"] for p in nominal]
    after = [metrics["calibration_after"][f"coverage_{int(p*100)}"] for p in nominal]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.3, label="Perfect calibration")
    ax.plot(nominal, before, "o-", color="#EF5350", lw=2, markersize=8,
            label=f"Before temperature scaling (ECE={metrics['calibration_before']['ece']:.3f})")
    ax.plot(nominal, after, "o-", color="#43A047", lw=2, markersize=8,
            label=f"After temperature scaling  (ECE={metrics['calibration_after']['ece']:.3f}, "
                  f"T={metrics['temperature_calibration']:.2f})")

    for x, y in zip(nominal, before):
        ax.annotate(f"{y*100:.0f}%", (x, y), textcoords="offset points", xytext=(8, -12), fontsize=8, color="#EF5350")
    for x, y in zip(nominal, after):
        ax.annotate(f"{y*100:.0f}%", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8, color="#43A047")

    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Observed empirical coverage")
    ax.set_title("Reliability diagram -- Bayesian BiLSTM epistemic uncertainty")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    save_path = str(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_uncertainty_vs_error(preds, save_path, n_bins: int = 3):
    """Bins the test set into n_bins groups by predicted (calibrated)
    uncertainty and plots actual MAE per bin. If the uncertainty
    estimate is meaningful, MAE should increase with predicted sigma --
    this is the quantified 'operational benefit' evidence a point-
    estimate model (RF/XGBoost/GB) has no equivalent for."""
    from scipy.stats import spearmanr

    y_true, y_pred = preds["y_true"], preds["y_pred"]
    sigma = preds["y_pred_std_calibrated"] if "y_pred_std_calibrated" in preds else preds["y_pred_std"]
    abs_err = np.abs(y_pred - y_true)

    bin_labels = ["Low"] + (["Medium"] if n_bins == 3 else []) + ["High"]
    bin_edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(sigma, bin_edges[1:-1]), 0, n_bins - 1)

    bin_mae = [abs_err[bin_idx == i].mean() for i in range(n_bins)]
    bin_mean_sigma = [sigma[bin_idx == i].mean() for i in range(n_bins)]

    rho, pval = spearmanr(sigma, abs_err)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(bin_labels, bin_mae, color=["#43A047", "#FB8C00", "#E53935"][:n_bins], alpha=0.88)
    for i, v in enumerate(bin_mae):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("MAE (actual error)")
    axes[0].set_xlabel("Predicted uncertainty tertile")
    axes[0].set_title("Actual error by predicted-uncertainty tier")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].scatter(sigma, abs_err, s=14, alpha=0.35, color="#5C6BC0")
    axes[1].set_xlabel("Predicted uncertainty (calibrated sigma)")
    axes[1].set_ylabel("Actual absolute error")
    axes[1].set_title(f"Spearman rho={rho:.3f} (p={pval:.2e})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = str(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    mae_increase_pct = 100 * (bin_mae[-1] - bin_mae[0]) / bin_mae[0] if bin_mae[0] > 0 else float("nan")
    return {
        "bin_labels": bin_labels,
        "bin_mae": [float(v) for v in bin_mae],
        "bin_mean_sigma": [float(v) for v in bin_mean_sigma],
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
        "mae_increase_high_vs_low_pct": float(mae_increase_pct),
    }, save_path


def run() -> dict:
    """Builds both figures + a summary from the Bayesian BiLSTM's
    already-saved results. Raises a clear error if it hasn't been
    trained yet. Standalone: python -m src.evaluation.uncertainty_analysis"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, preds = _load_bilstm_results()

    reliability_path = plot_reliability_diagram(metrics, OUT_DIR / "reliability_diagram.png")
    logger.info(f"Saved reliability diagram -> {reliability_path}")

    unc_summary, unc_path = plot_uncertainty_vs_error(preds, OUT_DIR / "uncertainty_vs_error.png")
    logger.info(f"Saved uncertainty-vs-error plot -> {unc_path}")
    logger.info(f"MAE in highest-uncertainty tertile is {unc_summary['mae_increase_high_vs_low_pct']:.1f}% "
                f"higher than in the lowest-uncertainty tertile "
                f"(Spearman rho={unc_summary['spearman_rho']:.3f}, p={unc_summary['spearman_pvalue']:.2e})")

    result = {
        "calibration_before_ece": metrics["calibration_before"]["ece"],
        "calibration_after_ece": metrics["calibration_after"]["ece"],
        "temperature": metrics["temperature_calibration"],
        **unc_summary,
    }
    save_json(OUT_DIR / "summary.json", result)
    return result


if __name__ == "__main__":
    from src.utils.logging_setup import get_logger
    get_logger()
    run()
