"""
dashboard/pages/evaluations.py
--------------------------------
Evaluations page — reads whatever result files exist on disk and
displays them. Shows nothing (with a clear message) if the pipeline
hasn't produced outputs yet. Never imports TensorFlow.
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _metric_color(value: float, metric: str) -> str:
    """Return a colour based on whether the value is good/bad."""
    if metric in ("mae", "rmse"):
        return "#86efac" if value < 0.05 else ("#fbbf24" if value < 0.15 else "#f87171")
    if metric == "r2":
        return "#86efac" if value > 0.85 else ("#fbbf24" if value > 0.65 else "#f87171")
    return "#e2e8f0"


def _no_data_banner(message: str):
    st.markdown(
        f'<div style="background:#161b27;border:1px solid #1e2736;border-left:3px solid #f59e0b;'
        f'border-radius:6px;padding:16px 20px;color:#94a3b8;font-size:0.85rem;">'
        f'⚠ &nbsp;{message}</div>',
        unsafe_allow_html=True,
    )


# ── collect what exists ───────────────────────────────────────────────────────

available_models = []
model_data       = {}

for mname in config.MODEL_NAMES:
    metrics_path = config.RESULTS_DIR / mname / "metrics.json"
    preds_path   = config.RESULTS_DIR / mname / "predictions.npz"
    history_path = config.RESULTS_DIR / mname / "history.json"
    if metrics_path.exists():
        available_models.append(mname)
        model_data[mname] = {
            "metrics":  _load_json(metrics_path),
            "has_preds": preds_path.exists(),
            "history":  _load_json(history_path) if history_path.exists() else None,
            "label":    config.MODEL_LABELS[mname],
            "color":    config.MODEL_COLORS[mname],
        }

comparison_csv    = config.COMPARISON_DIR / "metrics_summary.csv"
comparison_master = config.COMPARISON_DIR / "model_comparison_master.png"
comparison_grid   = config.COMPARISON_DIR / "model_comparison_grid.png"
bootstrap_json    = config.ROUTING_DATA_DIR / "bootstrap_summary.json"
routing_perf_json = config.ROUTING_DATA_DIR / "routing_performance.json"
unc_dir           = config.RESULTS_DIR / "uncertainty_analysis"

# ── layout ────────────────────────────────────────────────────────────────────

if not available_models:
    st.markdown("## Evaluations")
    _no_data_banner(
        "No model results found. Train at least one model first: "
        "<code>python main.py --stage train --models bilstm</code>"
    )
    st.stop()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_metrics, tab_compare, tab_uncertainty, tab_bootstrap = st.tabs([
    "Model Metrics", "Comparison Figures", "Uncertainty Analysis", "Bootstrap CI"
])

# ══ TAB 1: per-model metrics ══════════════════════════════════════════════════
with tab_metrics:
    st.markdown('<div class="section-label">Test-set performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#64748b;font-size:0.8rem;margin-bottom:16px;">'
        f'Trained on <code style="color:#94a3b8">{config.TARGET_COL}</code> '
        f'· {len(available_models)}/{len(config.MODEL_NAMES)} models trained</div>',
        unsafe_allow_html=True,
    )

    # Metrics grid — one column per model
    cols = st.columns(len(available_models))
    for col, mname in zip(cols, available_models):
        d = model_data[mname]
        m = d["metrics"] or {}
        with col:
            # model header
            st.markdown(
                f'<div style="font-weight:600;font-size:0.88rem;color:{d["color"]};'
                f'margin-bottom:8px;padding-bottom:6px;border-bottom:2px solid {d["color"]}40;">'
                f'{d["label"]}</div>',
                unsafe_allow_html=True,
            )
            for key, label in [("mae","MAE"), ("rmse","RMSE"), ("r2","R²")]:
                val = m.get(key)
                if val is not None:
                    clr = _metric_color(val, key)
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:5px 0;border-bottom:1px solid #1a2030;">'
                        f'<span style="color:#64748b;font-size:0.8rem;">{label}</span>'
                        f'<span style="color:{clr};font-family:\'JetBrains Mono\',monospace;'
                        f'font-size:0.85rem;font-weight:500;">{val:.4f}</span></div>',
                        unsafe_allow_html=True,
                    )

            # BiLSTM-specific: calibration
            if mname == "bayesian_bilstm":
                cal_after = m.get("calibration_after", {})
                ece = cal_after.get("ece")
                T   = m.get("temperature_calibration")
                sigma = m.get("mean_sigma")
                if ece is not None:
                    st.markdown(
                        f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #1e2736;">'
                        f'<div style="font-size:0.72rem;color:#475569;margin-bottom:4px;">Uncertainty</div>',
                        unsafe_allow_html=True,
                    )
                    for lbl, val in [("ECE (after calib)", ece), ("Temp. T", T), ("Mean σ", sigma)]:
                        if val is not None:
                            st.markdown(
                                f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
                                f'<span style="color:#64748b;font-size:0.76rem;">{lbl}</span>'
                                f'<span style="color:#94a3b8;font-family:\'JetBrains Mono\',monospace;'
                                f'font-size:0.76rem;">{val:.4f}</span></div>',
                                unsafe_allow_html=True,
                            )
                    st.markdown("</div>", unsafe_allow_html=True)

    # Training curves for neural models
    neural = [m for m in available_models if model_data[m]["history"] is not None]
    if neural:
        st.markdown('<div class="section-label" style="margin-top:32px;">Training curves</div>', unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go
            selected_neural = st.selectbox("Model", neural,
                                           format_func=lambda m: model_data[m]["label"])
            hist = model_data[selected_neural]["history"]
            color = model_data[selected_neural]["color"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hist.get("loss",[]), name="Train loss",
                                     line=dict(color=color, width=2)))
            fig.add_trace(go.Scatter(y=hist.get("val_loss",[]), name="Val loss",
                                     line=dict(color=color, width=2, dash="dash")))
            fig.update_layout(
                template="plotly_dark", height=280,
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                margin=dict(l=0, r=0, t=16, b=0),
                legend=dict(font=dict(size=11)),
                xaxis_title="Epoch", yaxis_title="Huber loss",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly for interactive training curves: `pip install plotly`")


# ══ TAB 2: comparison figures ════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-label">Cross-model comparison</div>', unsafe_allow_html=True)

    if comparison_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(comparison_csv)
            st.dataframe(
                df.style.highlight_min(subset=["mae","rmse"], color="#14532d")
                        .highlight_max(subset=["r2"], color="#14532d")
                        .format({"mae":"{:.4f}","rmse":"{:.4f}","r2":"{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.warning(f"Could not load metrics_summary.csv: {e}")
    else:
        _no_data_banner("Run <code>python main.py --stage compare</code> to generate the comparison table.")

    if comparison_master.exists():
        st.markdown('<div class="section-label" style="margin-top:24px;">Master comparison figure</div>',
                    unsafe_allow_html=True)
        st.image(str(comparison_master), use_container_width=True)

    if comparison_grid.exists():
        st.markdown('<div class="section-label" style="margin-top:24px;">Per-model grid</div>',
                    unsafe_allow_html=True)
        st.image(str(comparison_grid), use_container_width=True)

    if not comparison_csv.exists() and not comparison_master.exists():
        pass  # banner already shown above


# ══ TAB 3: uncertainty analysis ══════════════════════════════════════════════
with tab_uncertainty:
    st.markdown('<div class="section-label">Bayesian BiLSTM — uncertainty quality</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#64748b;font-size:0.8rem;margin-bottom:16px;">'
        'Answers the reviewer question: "why prefer this model despite lower point accuracy?" — '
        'if the model\'s uncertainty estimate is meaningful, predictions should be measurably '
        'worse exactly where it says it\'s unsure.</div>',
        unsafe_allow_html=True,
    )

    unc_summary = unc_dir / "summary.json"
    rel_diagram = unc_dir / "reliability_diagram.png"
    unc_plot    = unc_dir / "uncertainty_vs_error.png"

    if not unc_summary.exists():
        _no_data_banner(
            "Run uncertainty analysis first: "
            "<code>python -m src.evaluation.uncertainty_analysis</code>"
        )
    else:
        data = _load_json(unc_summary) or {}

        c1, c2, c3, c4 = st.columns(4)
        for col, (label, key, fmt) in zip(
            [c1, c2, c3, c4],
            [
                ("ECE before",    "calibration_before_ece",         ".4f"),
                ("ECE after",     "calibration_after_ece",          ".4f"),
                ("Temperature T", "temperature",                    ".4f"),
                ("MAE lift high→low", "mae_increase_high_vs_low_pct", ".1f%"),
            ],
        ):
            val = data.get(key)
            if val is not None:
                col.metric(label, f"{val:{fmt.lstrip('.')}}")

        # Spearman
        rho = data.get("spearman_rho")
        pval = data.get("spearman_pvalue")
        if rho is not None:
            direction = "✓ uncertainty correlates with actual error" if rho > 0 else "✗ no correlation"
            clr = "#86efac" if rho > 0.2 else "#f87171"
            st.markdown(
                f'<div style="margin:16px 0;padding:12px 16px;background:#161b27;'
                f'border-radius:8px;border:1px solid #1e2736;">'
                f'<span style="color:#64748b;font-size:0.8rem;">Spearman ρ(σ, |error|) = </span>'
                f'<span style="color:{clr};font-family:\'JetBrains Mono\',monospace;">{rho:.3f}</span>'
                f'<span style="color:#475569;font-size:0.78rem;margin-left:8px;">p={pval:.2e} · {direction}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        cols = st.columns(2)
        if rel_diagram.exists():
            with cols[0]:
                st.image(str(rel_diagram), caption="Reliability diagram")
        if unc_plot.exists():
            with cols[1]:
                st.image(str(unc_plot), caption="Uncertainty vs error")


# ══ TAB 4: bootstrap CI ══════════════════════════════════════════════════════
with tab_bootstrap:
    st.markdown('<div class="section-label">Statistical risk-reduction validation</div>',
                unsafe_allow_html=True)

    if not bootstrap_json.exists():
        _no_data_banner(
            "Run <code>python main.py --stage route</code> to generate bootstrap results."
        )
    else:
        bs = _load_json(bootstrap_json) or {}
        sig = bs.get("significant", False)
        sig_label = "Significant (CI does not cross zero)" if sig else "Not significant (CI crosses zero)"
        sig_color = "#86efac" if sig else "#f87171"

        st.markdown(
            f'<div style="padding:14px 18px;background:#161b27;border-radius:8px;'
            f'border:1px solid #1e2736;border-left:3px solid {sig_color};margin-bottom:20px;">'
            f'<span style="color:{sig_color};font-weight:600;font-size:0.9rem;">{sig_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean risk reduction", f"{bs.get('mean_reduction_pct', 0):.1f}%")
        c2.metric("95% CI low",  f"{bs.get('ci_95_lo', 0):.1f}%")
        c3.metric("95% CI high", f"{bs.get('ci_95_hi', 0):.1f}%")
        c4.metric("P(reduction > 0)", f"{bs.get('p_positive', 0)*100:.0f}%")

        st.markdown(
            f'<div style="color:#475569;font-size:0.78rem;margin-top:8px;">'
            f'n={bs.get("n_pairs",0)} OD pairs · {bs.get("n_bootstrap_samples",0):,} bootstrap samples</div>',
            unsafe_allow_html=True,
        )

        # Bootstrap distribution histogram if CSV exists
        bootstrap_csv_path = config.BOOTSTRAP_CI_CSV
        if bootstrap_csv_path.exists():
            try:
                import pandas as pd, plotly.graph_objects as go
                boot_df = pd.read_csv(bootstrap_csv_path)
                col_name = boot_df.columns[0]
                vals = boot_df[col_name].dropna()

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=vals, nbinsx=80,
                    marker_color="#3b82f6", opacity=0.75,
                    name="Bootstrap samples",
                ))
                ci_lo = bs.get("ci_95_lo", 0)
                ci_hi = bs.get("ci_95_hi", 0)
                mean  = bs.get("mean_reduction_pct", 0)
                for xval, lbl, clr in [
                    (0,      "zero",    "#f87171"),
                    (mean,   "mean",    "#f59e0b"),
                    (ci_lo,  "CI 2.5%", "#94a3b8"),
                    (ci_hi,  "CI 97.5%","#94a3b8"),
                ]:
                    fig.add_vline(x=xval, line_color=clr, line_dash="dash",
                                  annotation_text=lbl, annotation_position="top",
                                  annotation_font_size=10)
                fig.update_layout(
                    template="plotly_dark", height=300,
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    margin=dict(l=0, r=0, t=24, b=0),
                    xaxis_title="Bootstrapped mean risk reduction (%)",
                    yaxis_title="Count",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for the bootstrap histogram: `pip install plotly`")

    # Routing performance (also relevant here)
    if routing_perf_json.exists():
        perf = _load_json(routing_perf_json) or {}
        st.markdown('<div class="section-label" style="margin-top:28px;">Route computation time</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#64748b;font-size:0.78rem;margin-bottom:12px;">'
            'Time to compute all 5 route modes for one OD pair using the pre-built routing map '
            '(not training time).</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean",   f"{perf.get('mean_ms',0):.0f} ms")
        c2.metric("Median", f"{perf.get('median_ms',0):.0f} ms")
        c3.metric("p95",    f"{perf.get('p95_ms',0):.0f} ms")
        c4.metric("Max",    f"{perf.get('max_ms',0):.0f} ms")
