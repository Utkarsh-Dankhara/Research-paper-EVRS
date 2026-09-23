"""
dashboard/pages/route_explorer.py
-----------------------------------
Route Explorer — the existing routing + reliability viewer,
unchanged functionally, now living as a named page in the multi-page app.
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from src.routing.graph_utils import make_haversine, random_od_pair
from src.routing.reliability import (
    build_combined_map, build_edges_std, build_summary_report,
    compute_eta_ci, plot_reliability_profile,
)
from src.routing.router import prepare_routing_graphs, run_route


@st.cache_resource(show_spinner="Loading routing graphs and injecting Bayesian weights (once per session)...")
def _load_routing_context():
    return prepare_routing_graphs()


def _reset_od_state():
    st.session_state.od = None
    st.session_state.click_stage = "start"
    st.session_state.click_start = None


# ── page ──────────────────────────────────────────────────────────────────────

st.markdown(
    '<div style="color:#64748b;font-size:0.82rem;margin-bottom:16px;">'
    'Reads the pre-built routing map and city graphs — does not retrain anything. '
    'Run <code>python main.py --stage train --models bilstm</code> then '
    '<code>python main.py --stage route</code> first.</div>',
    unsafe_allow_html=True,
)

try:
    graphs, df_w, alphas, city_centers = _load_routing_context()
except FileNotFoundError as e:
    st.error(str(e))
    st.markdown(
        '<div style="margin-top:12px;padding:12px 16px;background:#1c1917;border-radius:8px;'
        'border-left:3px solid #f59e0b;font-size:0.82rem;color:#fbbf24;">'
        'Go to the <b>Pipeline Control</b> page to see which stages need to run.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if "city_code" not in st.session_state:
    st.session_state.city_code = next(iter(config.CITIES))
    _reset_od_state()

city_code = st.sidebar.selectbox(
    "City",
    options=list(config.CITIES.keys()),
    format_func=lambda c: config.CITIES[c].split(",")[0],
    index=list(config.CITIES.keys()).index(st.session_state.city_code),
)
if city_code != st.session_state.city_code:
    st.session_state.city_code = city_code
    _reset_od_state()

G      = graphs[city_code]
center = city_centers[city_code]

st.sidebar.markdown(
    '<div style="font-size:0.72rem;color:#475569;margin:16px 0 8px 0;'
    'text-transform:uppercase;letter-spacing:0.06em;">Origin / Destination</div>',
    unsafe_allow_html=True,
)
od_mode = st.sidebar.radio(
    "Pick how",
    ["Random pair", "Click on map", "Enter coordinates"],
    label_visibility="collapsed",
)

if od_mode == "Random pair":
    if st.sidebar.button("New random pair", use_container_width=True) or st.session_state.get("od") is None:
        rng = np.random.default_rng()
        s_lat, s_lng, e_lat, e_lng, label = random_od_pair(G, city_code, rng=rng)
        st.session_state.od = (s_lat, s_lng, e_lat, e_lng, label)

elif od_mode == "Enter coordinates":
    s_lat = st.sidebar.number_input("Start lat", value=center[0], format="%.6f")
    s_lng = st.sidebar.number_input("Start lon", value=center[1], format="%.6f")
    e_lat = st.sidebar.number_input("End lat",   value=center[0] + 0.01, format="%.6f")
    e_lng = st.sidebar.number_input("End lon",   value=center[1] + 0.01, format="%.6f")
    st.session_state.od = (s_lat, s_lng, e_lat, e_lng, "Manual coordinates")

else:
    import folium
    stage = st.session_state.get("click_stage", "start")
    st.sidebar.info("Click the map to set the **origin**." if stage == "start"
                    else "Click the map to set the **destination**.")
    if st.sidebar.button("Reset selection"):
        _reset_od_state(); stage = "start"

    pick_map = folium.Map(location=list(center), zoom_start=13, tiles="cartodbpositron")
    if st.session_state.get("click_start"):
        folium.Marker(st.session_state.click_start, tooltip="Origin",
                      icon=folium.Icon(color="green", icon="ambulance", prefix="fa")).add_to(pick_map)
    click = st_folium(pick_map, width=None, height=400, returned_objects=["last_clicked"])

    if click and click.get("last_clicked"):
        lat, lng = click["last_clicked"]["lat"], click["last_clicked"]["lng"]
        if stage == "start":
            st.session_state.click_start = (lat, lng)
            st.session_state.click_stage = "end"
            st.rerun()
        else:
            s_lat, s_lng = st.session_state.click_start
            st.session_state.od = (s_lat, s_lng, lat, lng, "Manually picked on map")
            st.session_state.click_stage = "start"
            st.session_state.click_start = None
            st.rerun()

    if st.session_state.get("od") is None:
        st.info("Pick an origin and destination on the map above to see routes.")
        st.stop()

s_lat, s_lng, e_lat, e_lng, od_label = st.session_state.od
st.sidebar.caption(od_label)

# ── routing ───────────────────────────────────────────────────────────────────
haversine    = make_haversine(G)
route_result = run_route(G, city_code, haversine, s_lat, s_lng, e_lat, e_lng)

if route_result is None:
    st.warning("Could not find a route between these two points. Try another pair.")
    st.stop()

edges_std = build_edges_std(route_result, df_w, mode="standard")
eta_ci    = compute_eta_ci(edges_std)
summary   = build_summary_report(edges_std, eta_ci, city_code, od_label)

col_map, col_stats = st.columns([2, 1])

with col_map:
    fmap = build_combined_map(route_result, edges_std, center,
                               (s_lat, s_lng), (e_lat, e_lng), od_label, alphas)
    st_folium(fmap, width=None, height=580, returned_objects=[])

with col_stats:
    st.metric("Mean segment reliability", f"{summary['mean_reliability_pct']:.1f}%")
    st.metric("ETA (standard route)",     summary["eta_point_estimate_fmt"])
    st.caption(f"95% CI: {summary['ci_95_lo_fmt']} – {summary['ci_95_hi_fmt']}")

    st.markdown(
        f'<div style="margin-top:16px;padding:10px 14px;background:#161b27;'
        f'border-radius:8px;border:1px solid #1e2736;">'
        f'<div style="color:#86efac;font-size:0.8rem;">High confidence &nbsp; '
        f'<b>{summary["high_confidence_segments"]}</b> / {summary["total_segments"]}</div>'
        f'<div style="color:#fbbf24;font-size:0.8rem;margin-top:4px;">Moderate uncertainty &nbsp; '
        f'<b>{summary["mid_confidence_segments"]}</b> / {summary["total_segments"]}</div>'
        f'<div style="color:#f87171;font-size:0.8rem;margin-top:4px;">High uncertainty &nbsp; '
        f'<b>{summary["low_confidence_segments"]}</b> / {summary["total_segments"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.72rem;color:#475569;margin:16px 0 6px 0;'
        'text-transform:uppercase;letter-spacing:0.06em;">Weakest segments</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        [{k: v for k, v in row.items() if k != "osmid_str"} for row in summary["weakest_segments"]],
        hide_index=True,
        use_container_width=True,
    )

st.markdown(
    '<div style="font-size:0.72rem;color:#475569;margin:20px 0 8px 0;'
    'text-transform:uppercase;letter-spacing:0.06em;">Segment reliability profile</div>',
    unsafe_allow_html=True,
)
profile_path = config.ROUTES_RELIABILITY_DIR / f"_dashboard_scratch_{city_code}.png"
plot_reliability_profile(edges_std, profile_path)
st.image(str(profile_path), use_container_width=True)
