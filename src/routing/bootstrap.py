"""
src/routing/bootstrap.py

Ported from Phase 4's bootstrap CI section of
notebooks/EVRS_NN_multi_models.ipynb: generates config.N_PAIRS_PER_CITY
random OD pairs per city, routes all of them, builds
route_comparison_metrics.csv, and bootstrap-resamples (config.
N_BOOTSTRAP_SAMPLES) the per-pair risk reductions into a 95% CI.

This is what statistically backs the risk-reduction claim -- it needs
to hold across all 3 cities, not just one hand-picked city.

Fully self-contained: loads/injects routing weights itself (via
src.routing.router.prepare_routing_graphs), does not depend on
src.routing.router.run() having been called first.

Saves data/routing/route_comparison_metrics.csv, data/routing/bootstrap_ci.csv
(raw bootstrap resample distribution), and data/routing/bootstrap_summary.json
(the actual mean/CI/significance conclusion, as a re-readable artifact
instead of only existing in the log text).

Standalone: python -m src.routing.bootstrap
"""
import logging

import numpy as np
import pandas as pd

import config
from src.routing.graph_utils import make_haversine, random_od_pair
from src.routing.router import prepare_routing_graphs, run_route
from src.utils.io import ensure_project_dirs, output_exists, save_json

logger = logging.getLogger(__name__)


def _generate_and_route_pairs(graphs: dict, n_pairs_per_city: int, rng):
    """Generates n_pairs_per_city random OD pairs per city and routes
    all of them in all 5 modes, timing each call to run_route() -- this
    is the number that matters for real-time deployment feasibility
    (computing a route at request time), which is a completely separate
    cost from one-time model TRAINING (MC-Dropout's many stochastic
    passes happen once, when the routing map is built -- not per route
    request; see the routing_performance.json this produces)."""
    import time

    all_results = []
    route_times_sec = []
    pair_id = 1
    for city_code, G in graphs.items():
        haversine = make_haversine(G)
        for _ in range(n_pairs_per_city):
            s_lat, s_lng, e_lat, e_lng, label = random_od_pair(G, city_code, rng=rng)
            t0 = time.time()
            res = run_route(G, city_code, haversine, s_lat, s_lng, e_lat, e_lng)
            elapsed = time.time() - t0
            if res:
                route_times_sec.append(elapsed)
                res["description"] = label
                res["pair_id"] = pair_id
                all_results.append(res)
                risk_red = (res["baseline"]["w_std"] - res["standard"]["w_std"]) / res["baseline"]["w_std"] * 100
                logger.info(f"  Pair {pair_id:2d} [{city_code}] ({label}): risk_red={risk_red:+.1f}%  "
                            f"std_nodes={res['standard']['nodes']}  cau_nodes={res['cautious']['nodes']}  "
                            f"({elapsed*1000:.0f} ms for all 5 modes)")
            else:
                logger.info(f"  Pair {pair_id:2d} [{city_code}] ({label}): SKIPPED (no route found)")
            pair_id += 1
    logger.info(f"{len(all_results)} pairs successfully routed across {len(graphs)} cities")
    return all_results, route_times_sec


def _save_routing_performance(route_times_sec: list):
    """Saves the per-route timing summary -- this is the concrete answer
    to 'report average route computation time and assess suitability
    for real-time emergency response' (routing itself, not training)."""
    if not route_times_sec:
        return None
    arr = np.asarray(route_times_sec)
    perf = {
        "n_routes_timed": len(arr),
        "mean_ms": float(arr.mean() * 1000),
        "median_ms": float(np.median(arr) * 1000),
        "p95_ms": float(np.percentile(arr, 95) * 1000),
        "max_ms": float(arr.max() * 1000),
        "note": ("Time to compute all 5 route modes (baseline/fastest/urgent/standard/cautious) "
                 "for one origin-destination pair, using the already-trained model and the "
                 "already-built routing map (data/routing/bayesian_routing_map.csv). Does NOT "
                 "include model training or the one-time full-graph MC-Dropout inference that "
                 "builds the routing map -- those happen once, offline, not per route request."),
    }
    save_json(config.ROUTING_DATA_DIR / "routing_performance.json", perf)
    logger.info(f"Routing performance: mean={perf['mean_ms']:.1f}ms  median={perf['median_ms']:.1f}ms  "
                f"p95={perf['p95_ms']:.1f}ms  max={perf['max_ms']:.1f}ms  (n={perf['n_routes_timed']})")
    return perf


def _build_comparison_table(all_results: list) -> pd.DataFrame:
    rows = []
    for res in all_results:
        base = res["baseline"]
        for mode in ["fastest", "urgent", "standard", "cautious"]:
            m = res[mode]
            risk_red = (base["w_std"] - m["w_std"]) / base["w_std"] * 100 if base["w_std"] > 0 else 0
            dist_inc = (m["distance"] - base["distance"]) / base["distance"] * 100 if base["distance"] > 0 else 0
            rows.append({
                "Pair": res["pair_id"], "City": res["city_code"], "Route": mode,
                "Nodes": m["nodes"], "Distance (m)": round(m["distance"], 1),
                "Travel time (s)": round(m["travel_t"], 1),
                "Risk reduction %": round(risk_red, 1),
                "Distance overhead %": round(dist_inc, 1),
            })
    metrics_df = pd.DataFrame(rows)
    config.ROUTING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(config.ROUTE_COMPARISON_METRICS_CSV, index=False)
    logger.info(f"Saved -> {config.ROUTE_COMPARISON_METRICS_CSV}")

    summary = metrics_df.groupby("Route")[
        ["Risk reduction %", "Distance overhead %", "Travel time (s)"]
    ].agg(["mean", "std"]).round(2)
    logger.info(f"Route comparison summary (averaged over all OD pairs, all cities):\n{summary.to_string()}")

    summary_by_city = metrics_df[metrics_df["Route"] == "standard"].groupby("City")["Risk reduction %"] \
        .agg(["mean", "std"]).round(2)
    logger.info(f"Standard-route risk reduction by city:\n{summary_by_city.to_string()}")
    return metrics_df


def _bootstrap_ci(all_results: list, n_boot: int, rng) -> dict:
    per_pair_reductions = []
    for res in all_results:
        base_risk = res["baseline"]["w_std"]
        std_risk = res["standard"]["w_std"]
        if base_risk > 0:
            per_pair_reductions.append((base_risk - std_risk) / base_risk * 100)
    per_pair_reductions = np.array(per_pair_reductions)

    boot_means = np.array([
        rng.choice(per_pair_reductions, size=len(per_pair_reductions), replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    p_positive = float((per_pair_reductions > 0).mean())

    pd.DataFrame({"boot_risk_reduction": boot_means}).to_csv(config.BOOTSTRAP_CI_CSV, index=False)
    logger.info(f"Saved -> {config.BOOTSTRAP_CI_CSV}")

    result = {
        "mean_reduction_pct": float(per_pair_reductions.mean()),
        "std_reduction_pct": float(per_pair_reductions.std()),
        "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
        "p_positive": p_positive,
        "significant": ci_lo > 0,
        "n_pairs": len(per_pair_reductions),
        "n_bootstrap_samples": n_boot,
    }
    save_json(config.ROUTING_DATA_DIR / "bootstrap_summary.json", result)
    logger.info("=" * 58)
    logger.info("  BOOTSTRAP CI - Risk Reduction (Standard vs Baseline)")
    logger.info("=" * 58)
    logger.info(f"  Mean reduction : {result['mean_reduction_pct']:.1f}%  +/-{result['std_reduction_pct']:.1f}%")
    logger.info(f"  95% CI         : {ci_lo:.1f}% - {ci_hi:.1f}%")
    logger.info(f"  P(reduction>0) : {p_positive * 100:.0f}%  (>=95% = significant)")
    logger.info("  Significant -- CI does not cross zero" if result["significant"]
                else "  Not significant -- consider more OD pairs")
    logger.info("=" * 58)
    return result


def run(force: bool = False) -> dict:
    """
    Statistical validation stage: generates config.N_PAIRS_PER_CITY
    random OD pairs per city, routes all of them, builds
    route_comparison_metrics.csv, and bootstraps a 95% CI on the
    standard route's risk reduction vs baseline.

    Skips entirely if bootstrap_ci.csv already exists and force=False.
    """
    summary_path = config.ROUTING_DATA_DIR / "bootstrap_summary.json"
    if output_exists(config.BOOTSTRAP_CI_CSV) and not force:
        logger.info(f"Skipping bootstrap stage -- {config.BOOTSTRAP_CI_CSV} already exists "
                    f"(pass force=True to rerun).")
        if output_exists(summary_path):
            from src.utils.io import load_json
            return load_json(summary_path)
        return {}

    ensure_project_dirs()
    graphs, df_w, alphas, city_centers = prepare_routing_graphs(force=force)

    # FIX: two independent RNGs — OD generation and bootstrap resampling
    # must not share state. If some OD pairs fail routing, the number of
    # rng.choice() calls in _generate_and_route_pairs varies, so the
    # bootstrap resamples would be non-reproducible. Separate seeds keep
    # both stages fully deterministic and independent of each other.
    od_rng   = np.random.default_rng(config.SEED)
    boot_rng = np.random.default_rng(config.SEED + 1)

    all_results, route_times_sec = _generate_and_route_pairs(graphs, config.N_PAIRS_PER_CITY, od_rng)
    _save_routing_performance(route_times_sec)
    if not all_results:
        raise RuntimeError("No OD pairs could be routed in any city -- check graph connectivity.")

    _build_comparison_table(all_results)
    return _bootstrap_ci(all_results, config.N_BOOTSTRAP_SAMPLES, boot_rng)


if __name__ == "__main__":
    from src.utils.cli import parse_force_flag
    from src.utils.logging_setup import get_logger
    get_logger()
    run(force=parse_force_flag())
