"""
dashboard/app.py
----------------
EVRS multi-page dashboard entry point.

Pages:
  Pipeline Control  — status of every stage, command builder
  Evaluations       — metrics, plots, comparison tables from results/
  Route Explorer    — the existing routing map + reliability viewer

Launch: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="EVRS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* base */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2736;
}

/* nav links in sidebar */
[data-testid="stSidebarNav"] a {
    color: #94a3b8 !important;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
    padding: 6px 12px;
    border-radius: 6px;
    display: block;
    transition: background 0.15s;
}
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: #1e2d45;
    color: #f59e0b !important;
}

/* status pills */
.pill-ok      { background:#14532d; color:#86efac; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-family:'JetBrains Mono',monospace; }
.pill-missing { background:#1c1917; color:#78716c; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-family:'JetBrains Mono',monospace; }
.pill-warn    { background:#431407; color:#fdba74; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-family:'JetBrains Mono',monospace; }

/* command block */
.cmd-block {
    background: #0d1117;
    border: 1px solid #1e2736;
    border-left: 3px solid #f59e0b;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #f8fafc;
    margin: 8px 0 4px 0;
    word-break: break-all;
}

/* metric override */
[data-testid="stMetric"] {
    background: #161b27;
    border: 1px solid #1e2736;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #64748b; font-size: 0.78rem; }
[data-testid="stMetricValue"] { color: #f1f5f9; font-size: 1.4rem; font-weight: 600; }

/* section headers */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #475569;
    text-transform: uppercase;
    margin: 24px 0 10px 0;
}

/* table */
[data-testid="stDataFrame"] { border: 1px solid #1e2736; border-radius: 8px; }

/* hide default streamlit decoration */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── sidebar brand ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 16px 4px 24px 4px;">
        <div style="font-size:1.1rem; font-weight:600; color:#f1f5f9; letter-spacing:-0.01em;">
            EVRS
        </div>
        <div style="font-size:0.75rem; color:#475569; margin-top:2px;">
            Emergency Vehicle Routing System
        </div>
        <div style="margin-top:12px; padding-top:12px; border-top:1px solid #1e2736;
                    font-size:0.72rem; color:#334155; font-family:'JetBrains Mono',monospace;">
            Gandhinagar · Ahmedabad · Surat
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── pages ─────────────────────────────────────────────────────────────────────
pipeline_page   = st.Page("pages/pipeline.py",  title="Pipeline Control", icon="⚙️", default=True)
eval_page       = st.Page("pages/evaluations.py", title="Evaluations",    icon="📊")
route_page      = st.Page("pages/route_explorer.py", title="Route Explorer", icon="🗺️")

pg = st.navigation([pipeline_page, eval_page, route_page], position="sidebar")
pg.run()
