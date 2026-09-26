"""
src/routing/reliability.py

Ported from Phase 5 of notebooks/EVRS_NN_multi_models.ipynb:
  - build_edges_std() -- per-segment uncertainty/reliability attached to
    a routed edges GeoDataFrame
  - unc_to_color() -- green -> amber -> red confidence colormap
  - compute_eta_ci() -- ETA point estimate + 50%/95% confidence intervals
  - plot_reliability_profile() -- 3-panel chart (reliability %,
    epistemic sigma, cumulative risk cost)
  - build_combined_map() / build_reliability_only_map() -- the two
    folium maps (toggleable all-routes view, and confidence-focused view)
  - build_summary_report() -- high/mid/low confidence counts, ETA +
    weakest-3-segments report

Used by both the routing pipeline stage (src.routing.router.run(), one
representative route per city) and dashboard/app.py (any route the user
picks interactively).
"""
import logging

import folium
import matplotlib
matplotlib.use("Agg")  # headless-safe -- runs as a script, not in a notebook display
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.utils.io import save_json

logger = logging.getLogger(__name__)

ROUTE_STYLES = {
    "baseline": {"color": "#2196F3", "weight": 4, "opacity": 0.75, "dash_array": "10 5"},
    "fastest":  {"color": "#9C27B0", "weight": 4, "opacity": 0.75, "dash_array": "10 5"},
    "urgent":   {"color": "#43A047", "weight": 4, "opacity": 0.80, "dash_array": "5 3"},
    "cautious": {"color": "#FF6F00", "weight": 5, "opacity": 0.85, "dash_array": None},
    "standard": {"color": "#E53935", "weight": 6, "opacity": 0.90, "dash_array": None},
}
OTHER_ROUTE_STYLES = {
    "baseline": {"color": "#2196F3", "weight": 3, "opacity": 0.45, "dash_array": "8 4"},
    "fastest":  {"color": "#9C27B0", "weight": 3, "opacity": 0.40, "dash_array": "8 4"},
    "urgent":   {"color": "#43A047", "weight": 3, "opacity": 0.40, "dash_array": "5 3"},
    "cautious": {"color": "#FF6F00", "weight": 4, "opacity": 0.50, "dash_array": None},
}


def _route_labels(alphas: dict) -> dict:
    return {
        "baseline": "Shortest distance (baseline)",
        "fastest": "Fastest time",
        "urgent": f"Bayesian urgent  alpha={alphas['urgent']:.0f}  (time-only)",
        "cautious": f"Bayesian cautious  alpha={alphas['cautious']:.0f}  (risk-averse)",
        "standard": f"Bayesian standard  alpha={alphas['standard']:.0f}  (route colour)",
    }


# ============================================================
# Per-segment reliability
# ============================================================

def build_edges_std(route_result: dict, df_w: pd.DataFrame, mode: str = "standard"):
    """Attaches per-segment uncertainty/reliability to a routed edges
    GeoDataFrame, looked up against that route's own city (df_w is
    filtered by city_code first, since edge ids are only meaningful
    within their own city's graph)."""
    city_code = route_result["city_code"]
    df_w_city = df_w[df_w["city_code"] == city_code]

    unc_lookup = dict(zip(df_w_city["osmid_local"].astype(str), df_w_city["epistemic_uncertainty"]))
    cost_lookup = dict(zip(df_w_city["osmid_local"].astype(str), df_w_city["bayesian_base_cost"]))

    def flatten_eid(v):
        return str(v[0]) if isinstance(v, list) else str(v)

    edges = route_result[mode]["edges_gdf"].copy()
    edges["osmid_str"] = edges["osmid"].apply(flatten_eid)
    edges["uncertainty"] = edges["osmid_str"].map(unc_lookup).fillna(df_w_city["epistemic_uncertainty"].mean())
    edges["ai_base_cost"] = edges["osmid_str"].map(cost_lookup).fillna(df_w_city["bayesian_base_cost"].mean())

    sigma_max = df_w_city["epistemic_uncertainty"].max()
    edges["reliability"] = 1.0 - (edges["uncertainty"] / (sigma_max + 1e-9))
    u_min, u_max = edges["uncertainty"].min(), edges["uncertainty"].max()
    edges["uncertainty_norm"] = (edges["uncertainty"] - u_min) / (u_max - u_min + 1e-9)
    return edges


def unc_to_color(norm_val: float) -> str:
    cmap = mcolors.LinearSegmentedColormap.from_list("rel", ["#2ECC71", "#F39C12", "#E74C3C"])
    r, g, b, _ = cmap(float(np.clip(norm_val, 0, 1)))
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _fmt_seconds(s):
    s = max(0, s)
    return f"{int(s // 60)}m {int(s % 60):02d}s"


def compute_eta_ci(edges_std) -> dict:
    """Point-estimate ETA + 50%/95% confidence intervals, propagating
    per-segment epistemic uncertainty through to a route-level ETA sigma."""
    has_tt = "travel_time" in edges_std.columns
    tt_vals = edges_std["travel_time"].fillna(0).values if has_tt else np.ones(len(edges_std)) * 30
    tt_total = tt_vals.sum()

    sigma_cost_per_seg = edges_std["uncertainty"].values * tt_vals
    combined_sigma_cost = np.sqrt(np.sum(sigma_cost_per_seg ** 2))

    mean_risk_per_seg = (
        np.mean(edges_std["ai_base_cost"].values / (tt_vals + 1e-9))
        if "ai_base_cost" in edges_std.columns else 2.7
    )
    eta_sigma_sec = combined_sigma_cost / (mean_risk_per_seg + 1e-9)

    return {
        "eta_point_estimate_sec": float(tt_total),
        "eta_point_estimate_fmt": _fmt_seconds(tt_total),
        "eta_sigma_sec": float(eta_sigma_sec),
        "ci_50_lo_fmt": _fmt_seconds(tt_total - 0.674 * eta_sigma_sec),
        "ci_50_hi_fmt": _fmt_seconds(tt_total + 0.674 * eta_sigma_sec),
        "ci_95_lo_fmt": _fmt_seconds(tt_total - 1.960 * eta_sigma_sec),
        "ci_95_hi_fmt": _fmt_seconds(tt_total + 1.960 * eta_sigma_sec),
    }


# ============================================================
# Reliability profile chart
# ============================================================

def plot_reliability_profile(edges_std, save_path):
    """3-panel segment reliability profile: reliability %, epistemic
    sigma, cumulative risk cost along the route."""
    n = len(edges_std)
    segs = range(1, n + 1)
    reliabilities = edges_std["reliability"].values * 100
    uncertainties = edges_std["uncertainty"].values
    bar_colors = [unc_to_color(v) for v in edges_std["uncertainty_norm"].values]

    fig, axes = plt.subplots(3, 1, figsize=(max(12, n * 0.35), 11), gridspec_kw={"hspace": 0.55})
    fig.suptitle("Bayesian Standard Route - Segment Reliability Profile", fontsize=13, y=0.99)

    axes[0].bar(segs, reliabilities, color=bar_colors, edgecolor="none", width=0.75)
    axes[0].axhline(reliabilities.mean(), color="#333", ls="--", lw=1.2, label=f"Mean: {reliabilities.mean():.1f}%")
    axes[0].axhline(80, color="#888", ls=":", lw=0.8, label="80% threshold")
    axes[0].set_ylabel("Reliability (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Per-segment confidence  (green = reliable, red = uncertain)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_xticks(segs)

    thresh = np.percentile(uncertainties, 75)
    for i, (s, u) in enumerate(zip(segs, uncertainties)):
        if u >= thresh:
            axes[0].annotate(f"sigma={u:.2f}", xy=(s, reliabilities[i]), xytext=(s, reliabilities[i] + 3),
                              fontsize=8, ha="center", color="#B71C1C")

    axes[1].bar(segs, uncertainties, color=bar_colors, edgecolor="none", width=0.75)
    axes[1].axhline(uncertainties.mean(), color="#333", ls="--", lw=1.2, label=f"Mean sigma: {uncertainties.mean():.3f}")
    axes[1].set_ylabel("Epistemic uncertainty sigma")
    axes[1].set_title("Model uncertainty  (higher = model less confident about this road)", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_xticks(segs)

    cum_cost = np.cumsum(edges_std["ai_base_cost"].values)
    axes[2].fill_between(segs, cum_cost, alpha=0.25, color="#E53935")
    axes[2].plot(segs, cum_cost, color="#E53935", lw=2, marker="o", markersize=4)
    axes[2].set_ylabel("Cumulative risk cost")
    axes[2].set_xlabel("Segment index")
    axes[2].set_title("Risk exposure accumulation along route", fontsize=10)
    axes[2].grid(alpha=0.3)
    axes[2].set_xticks(segs)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved -> {save_path}")
    return save_path


# ============================================================
# Folium maps
# ============================================================

def _draw_route_polylines(route_result, mode, style, label, target):
    edges = route_result[mode]["edges_gdf"]
    for _, row in edges.iterrows():
        if row.geometry is None:
            continue
        coords = [(y, x) for x, y in row.geometry.coords]
        risk_val = row.get("w_standard", 0)
        tt_val = row.get("travel_time", 0)
        len_val = row.get("length", 0)
        popup_html = (
            f"<div style='font-family:sans-serif;font-size:13px;width:200px'>"
            f"<b>{label}</b><hr style='margin:4px 0'>"
            f"Length: <b>{len_val:.1f} m</b><br>"
            f"Travel time: <b>{tt_val:.1f} s</b><br>"
            f"Risk weight: <b>{risk_val:.2f}</b></div>"
        )
        kw = dict(locations=coords, color=style["color"], weight=style["weight"], opacity=style["opacity"],
                  tooltip=label, popup=folium.Popup(popup_html, max_width=220))
        if style.get("dash_array"):
            kw["dash_array"] = style["dash_array"]
        folium.PolyLine(**kw).add_to(target)


def _draw_confidence_overlay(edges_std, target, weight=8):
    for _, row in edges_std.iterrows():
        if row.geometry is None:
            continue
        coords = [(y, x) for x, y in row.geometry.coords]
        seg_color = unc_to_color(row["uncertainty_norm"])
        rel_pct = row["reliability"] * 100
        sigma_val = row["uncertainty"]
        tt_val = row.get("travel_time", 0) if "travel_time" in edges_std.columns else 0
        popup_html = (
            f"<div style='font-family:sans-serif;font-size:13px;width:220px'>"
            f"<b>Segment reliability</b>"
            f"<div style='background:#eee;border-radius:4px;overflow:hidden;margin:5px 0'>"
            f"<div style='width:{rel_pct:.0f}%;background:{seg_color};height:10px'></div></div>"
            f"Reliability: <b>{rel_pct:.1f}%</b><br>"
            f"Uncertainty sigma: <b>{sigma_val:.4f}</b><br>"
            f"Length: <b>{row.get('length', 0):.1f} m</b><br>"
            f"Travel time: <b>{tt_val:.1f} s</b><br>"
            f"AI base cost: <b>{row.get('ai_base_cost', 0):.2f}</b></div>"
        )
        folium.PolyLine(locations=coords, color=seg_color, weight=weight, opacity=0.95,
                         tooltip=f"Reliability: {rel_pct:.1f}% | sigma={sigma_val:.4f}",
                         popup=folium.Popup(popup_html, max_width=240)).add_to(target)


def _add_od_markers(fmap, start, end, od_label):
    start_lat, start_lng = start
    end_lat, end_lng = end
    folium.Marker([start_lat, start_lng], popup=folium.Popup(f"<b>Origin</b><br>{od_label}", max_width=180),
                  icon=folium.Icon(color="green", icon="ambulance", prefix="fa")).add_to(fmap)
    folium.Marker([end_lat, end_lng], popup=folium.Popup("<b>Destination</b>", max_width=200),
                  icon=folium.Icon(color="red", icon="plus", prefix="fa")).add_to(fmap)


def _add_full_legend(fmap, city_code, alphas, side="bottom:20px;right:20px"):
    html = f"""
<div id='evrs-legend' style='position:fixed;{side};z-index:1000;background:white;border-radius:8px;
            border:1px solid #ccc;font-family:sans-serif;box-shadow:0 2px 6px rgba(0,0,0,0.2);color:#111;'>
  <div onclick="var b=document.getElementById('evrs-legend-body');var open=b.style.display!=='none';
                b.style.display=open?'none':'block';
                document.getElementById('evrs-legend-arrow').innerHTML=open?'&#9650;':'&#9660;';"
       style='cursor:pointer;padding:6px 10px;font-weight:700;font-size:12px;color:#111;
              display:flex;justify-content:space-between;align-items:center;min-width:150px;'>
    <span>Legend</span><span id='evrs-legend-arrow'>&#9660;</span>
  </div>
  <div id='evrs-legend-body' style='display:none;padding:8px 10px 10px;font-size:11px;color:#111;
              border-top:1px solid #eee;min-width:190px;'>
    <p style='margin:0 0 4px;font-weight:700;font-size:11.5px;color:#111;'>Route modes - {city_code.upper()}</p>
    <span style='color:#2196F3;'>&#9135;&#9135;</span> Baseline<br>
    <span style='color:#9C27B0;'>&#9135;&#9135;</span> Fastest<br>
    <span style='color:#43A047;'>&#x2508;&#x2508;</span> Urgent (a={alphas['urgent']:.0f})<br>
    <span style='color:#FF6F00;'>&#9644;&#9644;</span> Cautious (a={alphas['cautious']:.0f})<br>
    <span style='color:#E53935;'>&#9644;&#9644;</span> Standard (a={alphas['standard']:.0f})<br>
    <p style='margin:6px 0 4px;font-weight:700;font-size:11.5px;color:#111;
              border-top:1px solid #eee;padding-top:4px;'>Confidence</p>
    <span style='color:#2ECC71;'>&#9644;&#9644;</span> High<br>
    <span style='color:#F39C12;'>&#9644;&#9644;</span> Moderate<br>
    <span style='color:#E74C3C;'>&#9644;&#9644;</span> Low<br>
  </div>
</div>"""
    fmap.get_root().html.add_child(folium.Element(html))


def _add_compact_legend(fmap, city_code, alphas, side="bottom:20px;left:20px"):
    html = f"""
<div id='evrs-legend-compact' style='position:fixed;{side};z-index:1000;background:white;border-radius:8px;
            border:1px solid #ccc;font-family:sans-serif;box-shadow:0 2px 6px rgba(0,0,0,0.2);color:#111;'>
  <div onclick="var b=document.getElementById('evrs-legend-compact-body');var open=b.style.display!=='none';
                b.style.display=open?'none':'block';
                document.getElementById('evrs-legend-compact-arrow').innerHTML=open?'&#9650;':'&#9660;';"
       style='cursor:pointer;padding:6px 10px;font-weight:700;font-size:12px;color:#111;
              display:flex;justify-content:space-between;align-items:center;min-width:150px;'>
    <span>Legend</span><span id='evrs-legend-compact-arrow'>&#9660;</span>
  </div>
  <div id='evrs-legend-compact-body' style='display:none;padding:8px 10px 10px;font-size:11px;color:#111;
              border-top:1px solid #eee;min-width:190px;'>
    <p style='margin:0 0 4px;font-weight:700;font-size:11.5px;color:#111;'>Confidence - {city_code.upper()}</p>
    <span style='color:#2ECC71;'>&#9644;</span> High confidence<br>
    <span style='color:#F39C12;'>&#9644;</span> Moderate uncertainty<br>
    <span style='color:#E74C3C;'>&#9644;</span> High uncertainty<br>
    <p style='margin:6px 0 4px;font-weight:700;font-size:11.5px;color:#111;
              border-top:1px solid #eee;padding-top:4px;'>Other routes (reference)</p>
    <span style='color:#2196F3;'>&#9135;</span> Baseline<br>
    <span style='color:#9C27B0;'>&#9135;</span> Fastest<br>
    <span style='color:#43A047;'>&#x2508;</span> Urgent (a={alphas['urgent']:.0f})<br>
    <span style='color:#FF6F00;'>&#9644;</span> Cautious (a={alphas['cautious']:.0f})<br>
  </div>
</div>"""
    fmap.get_root().html.add_child(folium.Element(html))


def build_combined_map(route_result, edges_std, city_center, start, end, od_label, alphas):
    """All 5 routes as toggleable layers + the standard route's
    confidence overlay. Matches the original notebook's
    {city}_combined_routes_and_reliability.html."""
    labels = _route_labels(alphas)
    default_on = {"baseline", "standard"}

    fmap = folium.Map(
        location=list(city_center),
        zoom_start=14,
        tiles=config.CARTO_TILE_URL,
        attr=config.CARTO_ATTR,
    )

    for mode in ["baseline", "fastest", "urgent", "cautious", "standard"]:
        if mode not in route_result:
            continue
        fg = folium.FeatureGroup(name=f"<- {labels[mode]}", show=(mode in default_on))
        _draw_route_polylines(route_result, mode, ROUTE_STYLES[mode], labels[mode], fg)
        fg.add_to(fmap)

    fg_conf = folium.FeatureGroup(
        name="Standard route - confidence coloring  (green=reliable, red=uncertain)", show=True)
    _draw_confidence_overlay(edges_std, fg_conf)
    fg_conf.add_to(fmap)

    _add_od_markers(fmap, start, end, od_label)
    _add_full_legend(fmap, route_result["city_code"], alphas)
    folium.LayerControl(collapsed=True, position="topleft").add_to(fmap)
    return fmap


def build_reliability_only_map(route_result, edges_std, city_center, start, end, od_label, alphas):
    """Other 4 routes shown as lighter, always-on reference lines + the
    standard route's confidence overlay as the focal layer. Matches the
    original notebook's {city}_route_reliability.html."""
    labels = _route_labels(alphas)

    fmap = folium.Map(
        location=list(city_center),
        zoom_start=14,
        tiles=config.CARTO_TILE_URL,
        attr=config.CARTO_ATTR,
    )
    for mode, style in OTHER_ROUTE_STYLES.items():
        if mode not in route_result:
            continue
        _draw_route_polylines(route_result, mode, style, labels[mode], fmap)

    _draw_confidence_overlay(edges_std, fmap)
    _add_od_markers(fmap, start, end, od_label)
    _add_compact_legend(fmap, route_result["city_code"], alphas)
    return fmap


# ============================================================
# Summary report
# ============================================================

def build_summary_report(edges_std, eta_ci: dict, city_code: str, od_label: str) -> dict:
    n = len(edges_std)
    n_hi = int((edges_std["reliability"] >= 0.80).sum())
    n_mid = int(((edges_std["reliability"] >= 0.50) & (edges_std["reliability"] < 0.80)).sum())
    n_lo = int((edges_std["reliability"] < 0.50).sum())
    weakest = edges_std.nlargest(3, "uncertainty")[["osmid_str", "uncertainty", "reliability", "length"]]

    return {
        "city": city_code, "od_label": od_label, "total_segments": n,
        "high_confidence_segments": n_hi, "mid_confidence_segments": n_mid, "low_confidence_segments": n_lo,
        "mean_reliability_pct": float(edges_std["reliability"].mean() * 100),
        **eta_ci,
        "weakest_segments": weakest.to_dict(orient="records"),
    }


# ============================================================
# Orchestrator -- everything for one already-routed city
# ============================================================

def build_city_route_outputs(route_result, df_w, city_centers, start_lat, start_lng, end_lat, end_lng,
                              od_label, alphas) -> dict:
    """Runs the full Phase 5 pipeline for one already-routed city: builds
    edges_std, ETA CI, both folium maps, the reliability profile chart,
    and the summary report -- then saves everything. Called once per
    city by src.routing.router.run(), and reusable as-is by
    dashboard/app.py for any route the user picks interactively."""
    city_code = route_result["city_code"]
    edges_std = build_edges_std(route_result, df_w, mode="standard")
    eta_ci = compute_eta_ci(edges_std)

    city_center = city_centers[city_code]
    config.ROUTES_MAPS_DIR.mkdir(parents=True, exist_ok=True)

    combined_map = build_combined_map(route_result, edges_std, city_center,
                                       (start_lat, start_lng), (end_lat, end_lng), od_label, alphas)
    combined_path = config.ROUTES_MAPS_DIR / f"{city_code}_combined_routes_and_reliability.html"
    combined_map.save(str(combined_path))
    logger.info(f"[{city_code}] Saved -> {combined_path}")

    reliability_map = build_reliability_only_map(route_result, edges_std, city_center,
                                                  (start_lat, start_lng), (end_lat, end_lng), od_label, alphas)
    reliability_path = config.ROUTES_MAPS_DIR / f"{city_code}_route_reliability.html"
    reliability_map.save(str(reliability_path))
    logger.info(f"[{city_code}] Saved -> {reliability_path}")

    config.ROUTES_RELIABILITY_DIR.mkdir(parents=True, exist_ok=True)
    plot_reliability_profile(edges_std, config.ROUTES_RELIABILITY_DIR / f"{city_code}_route_reliability_profile.png")

    summary = build_summary_report(edges_std, eta_ci, city_code, od_label)
    save_json(config.ROUTES_RELIABILITY_DIR / f"{city_code}_route_summary.json", summary)
    logger.info(f"[{city_code}] Mean reliability: {summary['mean_reliability_pct']:.1f}%  |  "
                f"ETA: {summary['eta_point_estimate_fmt']}  "
                f"(95% CI {summary['ci_95_lo_fmt']}-{summary['ci_95_hi_fmt']})")

    return summary
