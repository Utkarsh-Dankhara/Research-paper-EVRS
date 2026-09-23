"""
src/evaluation/metrics.py

Shared metric helpers used by every model module, so MAE/RMSE/R2 and
uncertainty calibration are computed identically everywhere instead of
each model re-deriving slightly different versions.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Standard point-prediction metrics, computed on real-scale (not
    scaler-transformed) values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def coverage(y_true, y_mean, y_std, z) -> float:
    """Fraction of true values falling inside [mean - z*std, mean + z*std]."""
    y_true, y_mean, y_std = np.asarray(y_true), np.asarray(y_mean), np.asarray(y_std)
    return float(np.mean((y_true >= y_mean - z * y_std) & (y_true <= y_mean + z * y_std)))


# z-scores for the standard 50/80/95% confidence intervals
COVERAGE_Z = {"50": 0.674, "80": 1.282, "95": 1.960}
COVERAGE_TARGET = {"50": 0.50, "80": 0.80, "95": 0.95}


def expected_calibration_error(y_true, y_mean, y_std) -> dict:
    """ECE at 50/80/95% CI -- mean absolute gap between nominal and actual
    coverage. Lower is better calibrated (0 = perfect)."""
    cov = {k: coverage(y_true, y_mean, y_std, z) for k, z in COVERAGE_Z.items()}
    ece = float(np.mean([abs(cov[k] - COVERAGE_TARGET[k]) for k in cov]))
    return {
        "coverage_50": cov["50"], "coverage_80": cov["80"], "coverage_95": cov["95"],
        "ece": ece,
    }


def fit_temperature_scale(y_true, y_mean, y_std) -> float:
    """Fits a scalar T that minimizes ECE by rescaling the predicted
    uncertainty (std_calibrated = std * T) -- post-hoc calibration of
    MC-Dropout epistemic uncertainty. Bounded search over T in
    [0.1, 5.0], same as the original notebook's Phase 2 calibration step.
    Fit on the validation set, then applied to test/full-dataset std."""
    from scipy.optimize import minimize_scalar

    def ece_for_T(T):
        return expected_calibration_error(y_true, y_mean, np.asarray(y_std) * T)["ece"]

    result = minimize_scalar(ece_for_T, bounds=(0.1, 5.0), method="bounded")
    return float(result.x)
