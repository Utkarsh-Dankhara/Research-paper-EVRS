"""
dashboard/pages/pipeline.py
----------------------------
Pipeline Control page.

Shows the status of every pipeline stage (what exists on disk, what's
stale, what's missing), lets you pick exactly what to run and which
models, and builds the correct `python main.py` command to copy-paste
into your terminal.

Does NOT run the pipeline itself — training takes hours, lives in the
terminal where you can see logs, handle Ctrl-C, and have a persistent
log file. This page is the control surface for that process.
"""
import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

# ── helpers ───────────────────────────────────────────────────────────────────

def _age_str(path: Path) -> str:
    """Human-readable file age."""
    if not path.exists():
        return ""
    age = time.time() - path.stat().st_mtime
    if age < 3600:
        return f"{int(age/60)}m ago"
    if age < 86400:
        return f"{age/3600:.1f}h ago"
    return f"{int(age/86400)}d ago"


def _pill(label: str, kind: str) -> str:
    return f'<span class="pill-{kind}">{label}</span>'


def _status(exists: bool, stale: bool = False) -> str:
    if not exists:
        return _pill("missing", "missing")
    if stale:
        return _pill("stale", "warn")
    return _pill("ready", "ok")


def _stage_card(title: str, description: str, check_paths: list[Path],
                stale_after_paths: list[Path] = None):
    """Render one stage row with status pill and description."""
    exists = all(p.exists() for p in check_paths)
    stale  = False
    if exists and stale_after_paths:
        # stale if any upstream path is newer than any of our outputs
        if check_paths:
            our_mtime = min(p.stat().st_mtime for p in check_paths if p.exists())
            upstream_mtime = max(
                (p.stat().st_mtime for p in stale_after_paths if p.exists()),
                default=0,
            )
            stale = upstream_mtime > our_mtime

    age = _age_str(check_paths[0]) if exists and check_paths else ""
    age_str = f'<span style="color:#475569;font-size:0.75rem;margin-left:8px;">{age}</span>' if age else ""

    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:10px;padding:10px 0;
                        border-bottom:1px solid #1e2736;">
            <div style="width:200px;font-weight:500;color:#cbd5e1;font-size:0.88rem;">{title}</div>
            <div style="flex:1;color:#64748b;font-size:0.82rem;">{description}</div>
            <div>{_status(exists, stale)}{age_str}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    return exists, stale


def _model_status(model_name: str) -> tuple[bool, str]:
    """Check whether a model artifact exists and return its MAE if available."""
    artifact_paths = {
        "bayesian_bilstm": config.MODELS_DIR / "bayesian_bilstm" / "model.keras",
        "random_forest":   config.MODELS_DIR / "random_forest"   / "model.joblib",
        "xgboost":         config.MODELS_DIR / "xgboost"         / "model.joblib",
        "vanilla_lstm":    config.MODELS_DIR / "vanilla_lstm"    / "model.keras",
        "transformer":     config.MODELS_DIR / "transformer"     / "model.keras",
    }
    exists = artifact_paths[model_name].exists()
    mae    = ""
    metrics_path = config.RESULTS_DIR / model_name / "metrics.json"
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text())
            mae = f"MAE {m.get('mae', '?'):.4f}"
        except Exception:
            pass
    return exists, mae


# ── page ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Pipeline Status</div>', unsafe_allow_html=True)

# --- Stage status table ---
processed_csv  = config.PROCESSED_CSV
sequences_npz  = config.SEQUENCES_DIR / "sequences.npz"
routing_map    = config.ROUTING_MAP_CSV
bootstrap_csv  = config.BOOTSTRAP_CI_CSV

_stage_card(
    "1 · Preprocess",
    "OSM extraction → feature engineering → risk_score_v2",
    [processed_csv],
)
_stage_card(
    "2 · Sequences",
    "Predecessor-chain sequences + RobustScaler + highway OHE",
    [sequences_npz],
    stale_after_paths=[processed_csv],
)

st.markdown('<div class="section-label" style="margin-top:20px;">Models</div>', unsafe_allow_html=True)

_MODEL_LABELS = {
    "bayesian_bilstm": "Bayesian BiLSTM  (primary)",
    "random_forest":   "Random Forest",
    "xgboost":         "XGBoost",
    "vanilla_lstm":    "Vanilla LSTM",
    "transformer":     "Transformer",
}
for mname, mlabel in _MODEL_LABELS.items():
    exists, mae = _model_status(mname)
    detail = mae if mae else "not trained"
    age = _age_str(config.MODELS_DIR / mname / ("model.keras" if "lstm" in mname or mname in ("bayesian_bilstm","transformer") else "model.joblib"))
    age_str = f'<span style="color:#475569;font-size:0.75rem;margin-left:8px;">{age}</span>' if age else ""
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:10px;padding:8px 0;
                        border-bottom:1px solid #1a2030;">
            <div style="width:200px;font-size:0.85rem;color:#94a3b8;">{mlabel}</div>
            <div style="flex:1;color:#475569;font-size:0.8rem;font-family:'JetBrains Mono',monospace;">{detail}</div>
            <div>{_status(exists)}{age_str}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-label" style="margin-top:20px;">Downstream</div>', unsafe_allow_html=True)

_stage_card(
    "Compare",
    "Cross-model metrics, figures, metrics_summary.csv",
    [config.COMPARISON_DIR / "metrics_summary.csv"],
    stale_after_paths=[config.RESULTS_DIR / m / "metrics.json" for m in config.MODEL_NAMES],
)
_stage_card(
    "Routing maps",
    "Representative A* routes + reliability charts per city",
    [config.ROUTING_MAP_CSV],
)
_stage_card(
    "Bootstrap CI",
    f"{config.N_PAIRS_PER_CITY} OD pairs/city → 95% CI on risk reduction",
    [bootstrap_csv],
    stale_after_paths=[routing_map],
)

# ── Command builder ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:32px;">Command Builder</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="color:#64748b;font-size:0.82rem;margin-bottom:16px;">'
    'Build the exact command to run. Copy it and paste it into your terminal — '
    'training runs there so you get live logs, Ctrl-C, and a persistent log file under logs/.'
    '</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 1])

with col1:
    stage = st.selectbox(
        "Stage",
        ["all", "preprocess", "sequences", "train", "compare", "route"],
        help="Which pipeline stage to run.",
    )

with col2:
    force = st.checkbox("--force  (ignore caches, rerun from scratch)", value=False)

model_selection = []
if stage in ("all", "train"):
    st.markdown(
        '<div style="font-size:0.8rem;color:#64748b;margin:10px 0 6px 0;">Models to train:</div>',
        unsafe_allow_html=True,
    )
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    _aliases = {"bayesian_bilstm": "bilstm", "random_forest": "rf",
                "xgboost": "xgb", "vanilla_lstm": "lstm", "transformer": "transformer"}
    with mcol1:
        if st.checkbox("BiLSTM", value=True, key="m_bilstm"):
            model_selection.append("bilstm")
    with mcol2:
        if st.checkbox("Random Forest", value=True, key="m_rf"):
            model_selection.append("rf")
    with mcol3:
        if st.checkbox("XGBoost", value=True, key="m_xgb"):
            model_selection.append("xgb")
    with mcol4:
        if st.checkbox("Vanilla LSTM", value=True, key="m_lstm"):
            model_selection.append("lstm")
    with mcol5:
        if st.checkbox("Transformer", value=True, key="m_trans"):
            model_selection.append("transformer")

# Build command
parts = ["python main.py", f"--stage {stage}"]
if force:
    parts.append("--force")
if model_selection and stage in ("all", "train"):
    all_selected = len(model_selection) == 5
    if not all_selected:
        parts.append(f"--models {','.join(model_selection)}")

cmd = " ".join(parts)

st.markdown(f'<div class="cmd-block">{cmd}</div>', unsafe_allow_html=True)
st.code(cmd, language=None)

# ── Log viewer ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:32px;">Recent Log</div>', unsafe_allow_html=True)

log_dir = config.LOGS_DIR
log_files = sorted(log_dir.glob("run_*.log"), reverse=True) if log_dir.exists() else []

if not log_files:
    st.markdown(
        '<div style="color:#475569;font-size:0.82rem;padding:16px 0;">'
        'No logs yet. Run the pipeline at least once to see logs here.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    selected_log = st.selectbox(
        "Log file",
        log_files,
        format_func=lambda p: p.name,
        label_visibility="collapsed",
    )
    lines = selected_log.read_text(errors="replace").splitlines()
    # Show last 120 lines; user can expand
    tail = lines[-120:]
    with st.expander(f"Showing last {len(tail)} lines of {selected_log.name}", expanded=True):
        st.code("\n".join(tail), language=None)
