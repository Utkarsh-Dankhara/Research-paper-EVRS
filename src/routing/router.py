"""
src/routing/router.py

Ported from Phase 4 of notebooks/EVRS_NN_multi_models.ipynb:
  - compute_alphas() -- calibrated alpha values so the 3 Bayesian modes
    produce genuinely different routes
  - make_weight_dict() / inject_weights() / inject_all_weights() -- with
    highway-mean-risk / global-mean-risk fallback for edges the model
    didn't cover
  - run_route() -- all 5 modes, with the A* heuristic=0 correctness fix
    for any weight other than 'length' (haversine is only an admissible
    heuristic in metres; using it for seconds/risk-cost weights broke
    A*'s optimality guarantee -- verified against Dijkstra on a test
    graph, off by 25-32%)
  - prepare_routing_graphs() -- shared setup (load routing map + graphs,
    compute alphas, inject weights) used by every routing consumer:
    this module's own run() stage, src.routing.bootstrap, and
    dashboard/app.py

Public run() is the "representative route per city" pipeline stage:
picks one random OD pair per city and produces the full reliability
visualization for it via src.routing.reliability.
"""
import logging

import networkx as nx
import numpy as np
import pandas as pd

import config
from src.data.loaders import load_city_graphs
from src.routing.graph_utils import compute_city_centers, make_haversine, random_od_pair
from src.utils.io import ensure_project_dirs, load_json, output_exists

logger = logging.getLogger(__name__)


# ============================================================
# Alpha calibration + weight injection
# ============================================================

def compute_alphas(df_w: pd.DataFrame) -> dict:
    """Calibrated alpha values so the 3 Bayesian routing modes produce
    genuinely different routes (not the same path with a relabeled cost):
      - urgent:   alpha = 0                          (time-only)
      - standard: sigma penalty = 20% of mean base cost at max sigma
      - cautious: sigma penalty = 60% of mean base cost at max sigma
    """
    mean_base_cost = df_w["bayesian_base_cost"].mean()
    max_uncertainty = df_w["epistemic_uncertainty"].max()
    alphas = {
        "urgent": config.ALPHA_URGENT,
        "standard": (config.ALPHA_STANDARD_FRACTION * mean_base_cost) / max_uncertainty,
        "cautious": (config.ALPHA_CAUTIOUS_FRACTION * mean_base_cost) / max_uncertainty,
    }
    logger.info(f"alpha_urgent={alphas['urgent']:.1f}  alpha_standard={alphas['standard']:.1f}  "
                f"alpha_cautious={alphas['cautious']:.1f}")
    return alphas


def make_weight_dict(df_w_city: pd.DataFrame, alpha: float) -> dict:
    costs = df_w_city["bayesian_base_cost"].to_numpy() + alpha * df_w_city["epistemic_uncertainty"].to_numpy()
    return dict(zip(df_w_city["osmid_local"].astype(str), costs))


def inject_weights(G, weight_dict: dict, attr_name: str, highway_fallback: dict, global_fallback: float):
    """Writes attr_name onto every edge in G: the model's predicted
    weight where available, or a highway-type-mean fallback (falling
    back further to the global mean) for any edge the model didn't
    cover. Returns (matched_count, fallback_count)."""
    matched = fallback = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        eid = data.get("osmid", "")
        eid = str(eid[0]) if isinstance(eid, list) else str(eid)
        if eid in weight_dict:
            data[attr_name] = weight_dict[eid]
            matched += 1
        else:
            hw = data.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0]
            tt = data.get("travel_time", data.get("length", 50) / (30 * 1000 / 3600))
            fallback_risk = highway_fallback.get(hw, global_fallback)
            data[attr_name] = fallback_risk * tt
            fallback += 1
    return matched, fallback


def inject_all_weights(graphs: dict, df_w: pd.DataFrame, alphas: dict,
                        highway_fallback: dict, global_fallback: float):
    """Injects w_urgent/w_standard/w_cautious into every city's graph.
    Weights are injected per city since each city's graph only knows
    about its own edges."""
    for city_code, G in graphs.items():
        df_w_city = df_w[df_w["city_code"] == city_code]
        for mode, alpha in alphas.items():
            wdict = make_weight_dict(df_w_city, alpha)
            m, f = inject_weights(G, wdict, f"w_{mode}", highway_fallback, global_fallback)
            if mode == "standard":
                total = m + f
                pct = (m / total * 100) if total else 0.0
                logger.info(f"[{city_code}] weight injection: {m:,} model edges ({pct:.1f}%) | "
                            f"{f:,} fallback edges ({100 - pct:.1f}%)")


def prepare_routing_graphs(force: bool = False):
    """
    Shared setup every routing consumer needs: loads the routing map
    (produced by src.models.bayesian_bilstm's Phase 3 full-graph
    inference), per-city graphs, and the highway-risk fallback table;
    computes the calibrated alphas; injects w_urgent/w_standard/
    w_cautious into every city's graph.

    Returns (graphs, df_w, alphas, city_centers).
    """
    if not output_exists(config.ROUTING_MAP_CSV):
        raise FileNotFoundError(
            f"{config.ROUTING_MAP_CSV} not found. Train the Bayesian BiLSTM first "
            f"(`python -m src.models.bayesian_bilstm`) -- its Phase 3 full-graph "
            f"inference step produces this file."
        )
    fallback_path = config.ROUTING_DATA_DIR / "highway_risk_fallback.json"
    if not output_exists(fallback_path):
        raise FileNotFoundError(
            f"{fallback_path} not found (but {config.ROUTING_MAP_CSV} exists -- inconsistent "
            f"routing cache). Both are produced together by training the Bayesian BiLSTM; "
            f"retrain with `python -m src.models.bayesian_bilstm --force` if you deleted one "
            f"of them manually."
        )

    df_w = pd.read_csv(config.ROUTING_MAP_CSV)

    # FIX: validate schema before attempting weight injection so a stale
    # routing map from the old pipeline gives a clear error, not a
    # confusing KeyError 20 lines later.
    _required = {"osmid_local", "city_code", "bayesian_base_cost", "epistemic_uncertainty"}
    _missing  = _required - set(df_w.columns)
    if _missing:
        raise ValueError(
            f"{config.ROUTING_MAP_CSV} is missing required columns: {_missing}. "
            f"This is likely a stale routing map from the old pipeline. "
            f"Retrain: python -m src.models.bayesian_bilstm --force"
        )

    logger.info(f"Loaded {len(df_w):,} road predictions across {df_w['city_code'].nunique()} cities")

    fallback = load_json(fallback_path)
    graphs = load_city_graphs(force_refresh=force)

    alphas = compute_alphas(df_w)
    inject_all_weights(graphs, df_w, alphas, fallback["by_highway"], fallback["global"])
    city_centers = compute_city_centers(graphs)

    return graphs, df_w, alphas, city_centers


# ============================================================
# Routing
# ============================================================

def run_route(G, city_code: str, haversine, start_lat, start_lng, end_lat, end_lng):
    """Runs all 5 routing modes (baseline/fastest/urgent/standard/
    cautious) between two coordinates within one city's graph, snapping
    each to the nearest graph node. Returns a results dict (including
    'city_code'), or None if no route exists (same node, or
    disconnected)."""
    try:
        import osmnx as ox

        sn = ox.distance.nearest_nodes(G, X=start_lng, Y=start_lat)
        en = ox.distance.nearest_nodes(G, X=end_lng, Y=end_lat)
        if sn == en:
            return None

        results = {}
        for mode, weight in [("baseline", "length"), ("fastest", "travel_time"),
                              ("urgent", "w_urgent"), ("standard", "w_standard"),
                              ("cautious", "w_cautious")]:
            # CORRECTNESS: haversine (metres) is only an admissible A* heuristic
            # for weight='length' (also metres). For any other weight (seconds,
            # or risk-cost units), drop the heuristic -- heuristic=0 is plain
            # Dijkstra, always optimal, negligible extra cost at this scale.
            heur = haversine if weight == "length" else (lambda u, v: 0)
            path = nx.astar_path(G, sn, en, heuristic=heur, weight=weight)
            edges = ox.routing.route_to_gdf(G, path)
            results[mode] = {
                "nodes": len(path),
                "distance": edges["length"].sum(),
                "travel_t": edges["travel_time"].sum() if "travel_time" in edges.columns else 0,
                "w_std": edges["w_standard"].sum() if "w_standard" in edges.columns else 0,
                "w_cau": edges["w_cautious"].sum() if "w_cautious" in edges.columns else 0,
                "edges_gdf": edges,
            }
        results["city_code"] = city_code
        return results
    except nx.NetworkXNoPath:
        logger.warning(f"[{city_code}] No path: ({start_lat:.4f},{start_lng:.4f}) → ({end_lat:.4f},{end_lng:.4f})")
        return None
    except nx.NodeNotFound as e:
        logger.warning(f"[{city_code}] Node not found: {e}")
        return None
    except Exception:
        # Genuine unexpected error — log full traceback so it's debuggable
        logger.exception(f"[{city_code}] Unexpected routing error")
        return None


# ============================================================
# Public entry point -- representative route per city
# ============================================================

def run(force: bool = False) -> dict:
    """
    Routing pipeline stage: for each city, picks one representative
    random OD pair, routes it in all 5 modes, and produces the full
    reliability visualization (combined map, reliability-only map,
    profile chart, summary report) via src.routing.reliability.

    Skips entirely if every city's combined map already exists and
    force=False.
    """
    already_done = all(
        output_exists(config.ROUTES_MAPS_DIR / f"{c}_combined_routes_and_reliability.html")
        for c in config.CITIES
    )
    if already_done and not force:
        logger.info("Skipping routing stage -- representative route maps already exist for every city "
                    "(pass force=True to rerun).")
        return {"cities": list(config.CITIES.keys())}

    from src.routing import reliability  # local import: avoids a reliability<->router import cycle

    ensure_project_dirs()
    graphs, df_w, alphas, city_centers = prepare_routing_graphs(force=force)

    rng = np.random.default_rng(config.SEED)
    results = {}
    for city_code, G in graphs.items():
        haversine = make_haversine(G)
        s_lat, s_lng, e_lat, e_lng, od_label = random_od_pair(G, city_code, rng=rng)
        logger.info(f"[{city_code}] {od_label}")

        route_result = run_route(G, city_code, haversine, s_lat, s_lng, e_lat, e_lng)
        if route_result is None:
            logger.warning(f"[{city_code}] Could not route representative pair -- skipping this city's maps.")
            continue

        results[city_code] = reliability.build_city_route_outputs(
            route_result, df_w, city_centers, s_lat, s_lng, e_lat, e_lng, od_label, alphas,
        )

    return results


if __name__ == "__main__":
    from src.utils.cli import parse_force_flag
    from src.utils.logging_setup import get_logger
    get_logger()
    run(force=parse_force_flag())
