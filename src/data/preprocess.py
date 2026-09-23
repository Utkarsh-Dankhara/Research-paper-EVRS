"""
src/data/preprocess.py

Ported from notebooks/dataset_full_preprocess.ipynb.

Multi-city OSM extraction (gnr/ahd/srt) -> single-pass cleaning ->
feature engineering (betweenness centrality, elevation stub,
missingness flags, per-city percentile normalization, density) ->
consolidated risk_score_v1 -> external-enrichment stubs -> final save.

Saves data/processed/roads_all_cities_processed.csv (+ .gpkg,
best-effort) and per-city graphs to data/raw/graph_cache/{city_code}.graphml
(via src.data.loaders, which also handles the download/cache logic).

Public entry point: run(force: bool = False) -> Path
"""
import logging

import networkx as nx
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

import config
from src.data.loaders import load_city_graphs
from src.utils.io import ensure_project_dirs

logger = logging.getLogger(__name__)


# ============================================================
# 1. Multi-city extraction
# ============================================================

def _extract_all_cities(graphs: dict):
    """Converts each city's graph to an edge GeoDataFrame, namespaces
    osmid/u/v so cities never collide, tags city/city_code, and
    concatenates into one combined edges_all frame. Also returns a
    per-city tag-completeness table (useful for spotting cities with
    much sparser OSM tagging than the others)."""
    import osmnx as ox

    city_edge_frames = []
    city_meta = []

    for city_code, place_name in config.CITIES.items():
        G = graphs[city_code]
        _, edges = ox.graph_to_gdfs(G)
        edges = edges.reset_index()  # brings u, v, key back as columns

        edges["city"] = place_name.split(",")[0]
        edges["city_code"] = city_code

        # Namespace osmid so the same OSM way id in two different cities never collides
        edges["osmid_raw"] = edges["osmid"]
        edges["osmid"] = edges["osmid"].apply(lambda x: x[0] if isinstance(x, list) else x)
        edges["osmid"] = edges.apply(lambda r: f"{city_code}_{r['osmid']}", axis=1)

        # Namespace node ids the same way so u/v stay unique across the combined graph
        edges["u"] = edges["u"].apply(lambda x: f"{city_code}_{x}")
        edges["v"] = edges["v"].apply(lambda x: f"{city_code}_{x}")

        city_edge_frames.append(edges)

        tag_cols = [c for c in ["maxspeed", "lanes", "surface", "lit", "width"] if c in edges.columns]
        city_meta.append({
            "city_code": city_code,
            "n_edges": len(edges),
            **{f"{c}_coverage_pct": round(100 * edges[c].notna().mean(), 1) for c in tag_cols},
        })

    edges_all = pd.concat(city_edge_frames, ignore_index=True)
    completeness_df = pd.DataFrame(city_meta)
    logger.info(f"Combined edges: {len(edges_all)} across {len(config.CITIES)} cities")
    logger.info(f"Per-city tag completeness:\n{completeness_df.to_string(index=False)}")
    return edges_all, completeness_df


# ============================================================
# 2. Single-pass cleaning
# ============================================================

def _normalize_highway(hw):
    """OSM 'highway' can be a list when a way was tagged with multiple values."""
    return hw[0] if isinstance(hw, list) else hw


def _clean_maxspeed(val):
    if isinstance(val, list):
        val = val[0]
    if isinstance(val, str):
        val = val.split()[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _clean_lanes(row):
    """Returns (lanes_value, was_estimated). Falls back to a highway-type
    default (config.DEFAULT_LANES) when lanes wasn't tagged or couldn't
    be parsed.

    NOTE: NaN/missing must be checked explicitly BEFORE the numeric
    parse -- float(nan) returns nan without raising, so a naive
    try/except float(val) silently lets missing values through as NaN
    instead of ever triggering the fallback estimate below."""
    val = row.get("lanes", np.nan)
    if isinstance(val, (list, np.ndarray)):
        val = val[0] if len(val) > 0 else np.nan
    if isinstance(val, str):
        val = val.split(";")[0]

    if not pd.isna(val):
        try:
            return float(val), False
        except (TypeError, ValueError):
            pass  # unparseable garbage string -- falls through to the estimate below

    base = config.DEFAULT_LANES.get(row.get("highway"), 2)
    if row.get("length", 0) > 1000:
        base += 1
    return float(base), True


def _parse_bool_like(val):
    """
    Handles OSM boolean-ish edge attributes (`reversed`, `oneway`), which can
    arrive as an actual bool, a real Python list of bools (straight from the
    graph), or a stringified list (e.g. after a CSV round-trip).
    """
    s = str(val).strip()
    if s.startswith("["):
        try:
            return int(any(e.strip() == "True" for e in s.strip("[]").split(",")))
        except (ValueError, TypeError):
            return 0
    return int(s == "True" or val is True)


def _compute_sinuosity(geom):
    """
    Ratio of actual path length to straight-line chord, computed in
    UTM 43N metres (EPSG:32643).

    FIX: Old version used np.linalg.norm on raw (lon, lat) degree
    coordinates. At Gujarat's latitude (~23°N): 1° lat ≈ 111 km,
    1° lon ≈ 102 km — a 9% scale difference. East-west roads had
    sinuosity systematically underestimated by up to ~15%. Projecting
    to UTM 43N (the correct metric CRS for Gujarat) gives true
    metre-scale distances before computing the ratio.
    """
    from pyproj import Transformer
    # always_xy=True: input is (lon, lat) matching Shapely coord order
    _proj = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)

    if geom is None:
        return 1.0
    try:
        coords = list(geom.coords)
    except AttributeError:
        try:
            coords = list(shapely_wkt.loads(str(geom)).coords)
        except (TypeError, ValueError, AttributeError):
            return 1.0

    if len(coords) < 2:
        return 1.0

    # Project each (lon, lat) → (easting_m, northing_m)
    proj_coords = np.array([_proj.transform(lon, lat) for lon, lat in coords])

    straight = np.linalg.norm(proj_coords[-1] - proj_coords[0])
    if straight < 1e-3:   # degenerate point-like geometry
        return 1.0

    path_len = sum(
        np.linalg.norm(proj_coords[i + 1] - proj_coords[i])
        for i in range(len(proj_coords) - 1)
    )
    return float(min(path_len / straight, 3.0))


def _estimate_junction(row, node_degree_map):
    if pd.notna(row.get("junction")):
        return 1
    deg_u = node_degree_map.get(row["u"], 0)
    deg_v = node_degree_map.get(row["v"], 0)
    return int(deg_u >= 3 or deg_v >= 3)


def _clean_dataframe(edges_all: pd.DataFrame, graphs: dict):
    """Single clean preprocessing pass: normalizes highway/maxspeed/lanes,
    computes travel_time/sinuosity/junction signals, filters out
    footway/pedestrian/track and sub-5m segments, and flags bidirectional
    duplicate pairs. Returns (df, node_degree)."""
    node_degree = {}
    for city_code, G in graphs.items():
        for n, d in G.degree():
            node_degree[f"{city_code}_{n}"] = d

    df = edges_all.copy()
    df["highway"] = df["highway"].apply(_normalize_highway)

    if "maxspeed" not in df.columns:
        df["maxspeed"] = np.nan
    df["maxspeed_clean"] = df["maxspeed"].apply(_clean_maxspeed)

    df["speed_kph"] = df.apply(
        lambda r: r["maxspeed_clean"] if pd.notna(r["maxspeed_clean"])
        else config.DEFAULT_SPEEDS.get(r["highway"], 30),
        axis=1,
    )
    df["travel_time"] = (df["length"] / 1000) / df["speed_kph"] * 3600

    lanes_result = df.apply(_clean_lanes, axis=1)
    df["lanes_estimated"] = lanes_result.apply(lambda t: t[0])
    df["lanes_was_estimated"] = lanes_result.apply(lambda t: int(t[1]))

    df["reversed"] = df["reversed"].apply(_parse_bool_like)

    if "oneway" not in df.columns:
        df["oneway"] = False
    df["oneway"] = df["oneway"].apply(_parse_bool_like)

    df["sinuosity"] = df["geometry"].apply(_compute_sinuosity)

    df["junction_estimated"] = df.apply(lambda r: _estimate_junction(r, node_degree), axis=1)
    df["junction_count"] = df.apply(
        lambda r: int(node_degree.get(r["u"], 0) >= 3) + int(node_degree.get(r["v"], 0) >= 3),
        axis=1,
    )
    df["avg_node_degree"] = df.apply(
        lambda r: (node_degree.get(r["u"], 1) + node_degree.get(r["v"], 1)) / 2.0,
        axis=1,
    )

    # ---------- filtering ----------
    df = df[~df["highway"].isin(["footway", "pedestrian", "track"])].copy()
    df = df[df["length"] >= 5.0].copy()

    # ---------- bidirectional duplicate flagging ----------
    # Keyed on the *unordered* node pair (not geometry equality -- a
    # bidirectional pair's reverse edge has coordinates in reverse order).
    # FIX: frozenset.__str__ is non-deterministic under Python hash
    # randomization (PYTHONHASHSEED differs per process). pd.Series
    # .duplicated() calls str() internally, so dedup results vary
    # between runs. sorted tuple is always ordered deterministically.
    df["_pair_key"] = list(zip(
        df["city_code"],
        df.apply(lambda r: tuple(sorted([str(r["u"]), str(r["v"])])), axis=1),
    ))
    df["is_bidirectional_duplicate"] = pd.Series(df["_pair_key"]).duplicated(keep=False).astype(int)
    df = df.drop(columns=["_pair_key"])

    logger.info(f"Rows after filtering + dedup flagging: {len(df)}")
    return df, node_degree


# ============================================================
# 3. Feature engineering
# ============================================================

def _collapse_to_digraph(G, weight="travel_time"):
    """
    Version-safe replacement for osmnx's multigraph->digraph helper. For
    each pair of parallel edges between the same u,v, keeps the one with
    the LOWEST weight, since that's the one a router would actually choose
    -- an arbitrary "last edge wins" collapse would bias centrality toward
    whichever parallel edge happened to be added last.
    """
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1.0)
        if D.has_edge(u, v):
            if w < D[u][v].get(weight, float("inf")):
                D[u][v].update(data)
        else:
            D.add_edge(u, v, **data)
    return D


def _add_betweenness_centrality(df: pd.DataFrame, graphs: dict) -> pd.DataFrame:
    """Per-city betweenness centrality (exact below
    config.EXACT_BETWEENNESS_THRESHOLD_NODES, sampled above it -- exact is
    O(V*E) and impractical on Ahmedabad/Surat-sized graphs), assigned to
    each edge as the mean of its endpoints' centrality."""
    import time

    edge_centrality_lookup = {}   # (city_code, osmid_raw) -> centrality value

    for city_code, G in graphs.items():
        D = _collapse_to_digraph(G, weight="travel_time")
        n_nodes = D.number_of_nodes()
        t0 = time.time()

        if n_nodes <= config.EXACT_BETWEENNESS_THRESHOLD_NODES:
            logger.info(f"[{city_code}] computing exact betweenness centrality "
                        f"({n_nodes} nodes) -- this can take a while on larger graphs...")
            node_bc = nx.betweenness_centrality(D, weight="travel_time", normalized=True)
        else:
            k = min(config.BETWEENNESS_SAMPLE_K, n_nodes)
            logger.info(f"[{city_code}] {n_nodes} nodes > "
                        f"{config.EXACT_BETWEENNESS_THRESHOLD_NODES} -- using k={k} "
                        f"sampled betweenness (approximate) instead of exact...")
            node_bc = nx.betweenness_centrality(
                D, k=k, weight="travel_time", normalized=True, seed=config.SEED
            )

        logger.info(f"[{city_code}] betweenness centrality done ({time.time() - t0:.1f}s)")

        for u, v, k_, data in G.edges(keys=True, data=True):
            raw_id = data.get("osmid")
            raw_id = raw_id[0] if isinstance(raw_id, list) else raw_id
            val = (node_bc.get(u, 0.0) + node_bc.get(v, 0.0)) / 2.0
            key = (city_code, str(raw_id))
            edge_centrality_lookup[key] = max(edge_centrality_lookup.get(key, 0.0), val)

    df["osmid_raw_str"] = df["osmid_raw"].apply(
        lambda x: str(x[0]) if isinstance(x, list) else str(x)
    )
    df["betweenness_centrality"] = df.apply(
        lambda r: edge_centrality_lookup.get((r["city_code"], r["osmid_raw_str"]), 0.0),
        axis=1,
    )
    return df


def _add_elevation(df: pd.DataFrame, graphs: dict) -> pd.DataFrame:
    """Elevation/grade enrichment -- documented stub. Skips cleanly (all
    NaN) unless config.ADD_ELEVATION=True and an ELEVATION_API_KEY
    environment variable is configured."""
    if not config.ADD_ELEVATION:
        df["grade_abs"] = np.nan
        logger.info("Elevation enrichment skipped (config.ADD_ELEVATION=False).")
        return df

    import os

    import osmnx as ox

    grade_lookup = {}
    for city_code, G in graphs.items():
        try:
            G = ox.elevation.add_node_elevations_google(G, api_key=os.environ.get("ELEVATION_API_KEY"))
            G = ox.elevation.add_edge_grades(G)
        except Exception as e:
            logger.warning(f"[{city_code}] elevation lookup skipped: {e}")
            continue
        for u, v, k, data in G.edges(keys=True, data=True):
            raw_id = data.get("osmid")
            raw_id = raw_id[0] if isinstance(raw_id, list) else raw_id
            grade_lookup[(city_code, str(raw_id))] = data.get("grade_abs", np.nan)

    df["grade_abs"] = df.apply(
        lambda r: grade_lookup.get((r["city_code"], r["osmid_raw_str"]), np.nan), axis=1
    )
    return df


def _add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Sparse OSM tags (surface/lit/width/maxspeed) get an explicit 'was
    this observed or defaulted' flag, instead of silently treating a
    filled-in default the same as a real observation."""
    for col in ["surface", "lit", "width", "maxspeed"]:
        flag_col = f"{col}_observed"
        if col in df.columns:
            df[flag_col] = df[col].notna().astype(int)
        else:
            df[col] = np.nan
            df[flag_col] = 0
    return df


def _add_percentile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-city percentile normalization, so e.g. 'high centrality' is
    comparable across cities of very different size/density instead of
    raw values dominated by the biggest city."""
    for col in ["betweenness_centrality", "sinuosity"]:
        df[f"{col}_pct_in_city"] = df.groupby("city_code")[col].rank(pct=True)
    return df


def _add_city_density(df: pd.DataFrame) -> pd.DataFrame:
    """City-level road density context feature (total edge length per
    city / city edge count, as a simple proxy)."""
    city_density = df.groupby("city_code")["length"].agg(["sum", "count"]).rename(
        columns={"sum": "city_total_road_length_m", "count": "city_edge_count"}
    )
    city_density["city_avg_edge_length_m"] = (
        city_density["city_total_road_length_m"] / city_density["city_edge_count"]
    )
    return df.merge(city_density, on="city_code", how="left")


# ============================================================
# 4. Consolidated risk formula (risk_score_v1)
# ============================================================

def _lane_risk_from_count(lanes):
    if pd.isna(lanes):
        return config.LANE_RISK_UNKNOWN
    try:
        lanes = int(float(lanes))
    except (TypeError, ValueError):
        return config.LANE_RISK_UNKNOWN
    if lanes in config.LANE_RISK_BY_COUNT:
        return config.LANE_RISK_BY_COUNT[lanes]
    if lanes <= 1:
        return config.LANE_RISK_DEFAULT
    return 0.9


def _compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Single consolidated risk formula -- every weight is named in
    config.RISK_WEIGHTS so it's auditable and easy to swap out once real
    accident data is available (see the Section 5 enrichment stubs)."""
    df["road_risk"] = df["highway"].map(config.ROAD_RISK).fillna(config.DEFAULT_ROAD_RISK)

    df["access_risk"] = df.get("access", pd.Series(np.nan, index=df.index)).apply(
        lambda x: config.ACCESS_RISK_PRIVATE if x == "private" else config.ACCESS_RISK_DEFAULT
    )

    df["lane_risk"] = df["lanes_estimated"].apply(_lane_risk_from_count)

    df["junction_risk"] = df["junction_estimated"].apply(
        lambda x: config.JUNCTION_RISK if x == 1 else config.NO_JUNCTION_RISK
    )

    df["speed_kph"] = pd.to_numeric(df["speed_kph"], errors="coerce").fillna(30).clip(lower=10)
    df["speed_risk"] = 50.0 / df["speed_kph"]

    # FIX: .notna() incorrectly flags bridge="no" as a bridge (notna("no")
    # is True). .isin() checks for explicit positive values only.
    _bridge = df["bridge"] if "bridge" in df.columns else pd.Series(np.nan, index=df.index)
    _tunnel = df["tunnel"] if "tunnel" in df.columns else pd.Series(np.nan, index=df.index)
    df["is_bridge"] = _bridge.isin(["yes", True, 1]).astype(int)
    df["is_tunnel"] = _tunnel.isin(["yes", True, 1]).astype(int)
    df["is_roundabout"] = (df.get("junction", pd.Series(np.nan, index=df.index)) == "roundabout").astype(int)

    # junction_risk is deliberately NOT included below -- junction_count
    # already captures that signal. Kept as its own column so a downstream
    # ML model can learn its own weight for it directly, rather than being
    # locked into this v1 heuristic.
    w = config.RISK_WEIGHTS
    df["risk_score_v1"] = (
        w["road"]             * df["road_risk"]
        + w["speed"]          * df["speed_risk"]
        + w["junction_count"] * df["junction_count"]
        + w["sinuosity"]      * (df["sinuosity"] - 1.0)
        + w["degree"]         * np.log1p(df["avg_node_degree"])
        + w["bridge"]         * df["is_bridge"]
        + w["tunnel"]         * df["is_tunnel"]
        + w["roundabout"]     * df["is_roundabout"]
        + w["lane"]           * df["lane_risk"]
        + w["access"]         * df["access_risk"]
        + w["centrality"]     * df["betweenness_centrality"]
    )

    # risk-aware routing cost: travel time scaled by risk, so a router can
    # trade "a bit longer" for "meaningfully safer" instead of always
    # taking the shortest path
    df["risk_aware_cost"] = df["travel_time"] * df["risk_score_v1"]
    return df


# ============================================================
# 5. External data enrichment (documented stubs)
# ============================================================

def _enrich_with_accidents(df: pd.DataFrame) -> pd.DataFrame:
    """Source: data.gov.in Road Accident datasets (free). Granularity is
    state/city/year -- NOT per-road -- so this can only give a city-level
    accident-rate prior, joined on `city`.
    TODO: download CSV -> data/external/accidents_<year>.csv, aggregate to
    a per-city rate, merge on `city`, then flip config.ENABLE_ACCIDENTS."""
    if not config.ENABLE_ACCIDENTS:
        df["city_accident_rate"] = np.nan
        return df
    # accidents = pd.read_csv(config.DATA_DIR / "external" / "accidents_2023.csv")
    # rate_by_city = accidents.groupby("city")["accidents_per_lakh_pop"].mean()
    # df["city_accident_rate"] = df["city"].map(rate_by_city)
    return df


def _enrich_with_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Source options (free/cheap tiers): TomTom Traffic Flow API, HERE
    Traffic API, or a self-hosted OSRM/Valhalla instance.
    TODO: query per-edge or per-bbox flow, join on osmid/geometry, then
    flip config.ENABLE_TRAFFIC."""
    if not config.ENABLE_TRAFFIC:
        df["live_congestion_factor"] = np.nan
        return df
    return df


def _enrich_with_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Source: OSM landuse/amenity tags (free, via ox.features_from_place)
    -- schools, hospitals, markets as proxies for higher-risk zones.
    TODO: pull tags={'amenity': ['school','hospital','marketplace']} per
    city, spatial-join to nearest edge within N meters, then flip
    config.ENABLE_ZONES."""
    if not config.ENABLE_ZONES:
        df["near_school_or_hospital"] = 0
        return df
    return df


def _enrich_with_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Source: Open-Meteo historical API (free, no key). Per-city
    coordinate + date range, then a simple city-level join.
    TODO: implement, then flip config.ENABLE_WEATHER."""
    if not config.ENABLE_WEATHER:
        df["avg_rainy_days_per_month"] = np.nan
        return df
    return df


# ============================================================
# 6. Final save
# ============================================================

def _finalize_and_save(df: pd.DataFrame):
    """Selects config.FINAL_COLUMNS (filling any missing ones with NaN),
    saves the CSV (always, geometry dropped -- not needed downstream,
    routing/reliability maps read geometry live from the OSM graph
    instead) and a GeoPackage (best-effort, needs geopandas)."""
    missing_cols = [c for c in config.FINAL_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Expected columns not found, filling with NaN: {missing_cols}")
        for c in missing_cols:
            df[c] = np.nan

    final_df = df[config.FINAL_COLUMNS].copy()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final_df.drop(columns=["geometry"]).to_csv(config.PROCESSED_CSV, index=False)
    logger.info(f"Saved CSV -> {config.PROCESSED_CSV} | rows={len(final_df)}")

    try:
        import geopandas as gpd
        gdf_out = gpd.GeoDataFrame(final_df, geometry="geometry", crs="EPSG:4326")
        gdf_out.to_file(config.PROCESSED_GPKG, driver="GPKG")
        logger.info(f"Saved GeoPackage -> {config.PROCESSED_GPKG}")
    except Exception as e:
        logger.warning(f"GeoPackage save skipped ({e}); CSV output above is still complete.")

    row_counts = final_df.groupby("city_code").size()
    logger.info(f"Rows per city:\n{row_counts.to_string()}")
    return config.PROCESSED_CSV


# ============================================================
# Public entry point
# ============================================================

def run(force: bool = False):
    """
    Runs the full multi-city preprocessing pipeline: extraction ->
    cleaning -> feature engineering -> risk_score_v1 -> external-
    enrichment stubs -> final save.

    Skips entirely (returns the existing path) if config.PROCESSED_CSV
    already exists and force=False -- this is the main runtime-saver on
    repeat runs. Passing force=True re-downloads every city from OSM
    (slow, especially Ahmedabad/Surat) and reprocesses from scratch.

    Returns the Path to the saved roads_all_cities_processed.csv.
    """
    if config.PROCESSED_CSV.exists() and not force:
        logger.info(f"Skipping preprocessing -- {config.PROCESSED_CSV} already exists "
                    f"(pass force=True to rerun).")
        return config.PROCESSED_CSV

    ensure_project_dirs()

    logger.info("Loading/downloading per-city OSM graphs (gnr/ahd/srt)...")
    graphs = load_city_graphs(force_refresh=force)

    logger.info("Extracting + combining edges across cities...")
    edges_all, completeness_df = _extract_all_cities(graphs)

    logger.info("Running single-pass cleaning...")
    df, node_degree = _clean_dataframe(edges_all, graphs)

    logger.info("Computing betweenness centrality (this is the slow step on larger cities)...")
    df = _add_betweenness_centrality(df, graphs)

    df = _add_elevation(df, graphs)
    df = _add_missingness_flags(df)
    df = _add_percentile_features(df)
    df = _add_city_density(df)

    # osmid_local: de-namespaced edge id needed by router.py + reliability.py
    df["osmid_local"] = df.apply(
        lambda r: str(r["osmid"])[len(str(r["city_code"])) + 1:], axis=1
    )

    logger.info("Computing risk_score_v1 (kept for backward comparison)...")
    df = _compute_risk_score(df)

    logger.info("Computing risk_score_v2 (structurally-grounded label used for training)...")
    from src.data.risk_v2 import compute_risk_score_v2, validate_label_quality
    df["risk_score_v2"] = compute_risk_score_v2(df)
    validate_label_quality(df)

    df = _enrich_with_accidents(df)
    df = _enrich_with_traffic(df)
    df = _enrich_with_zones(df)
    df = _enrich_with_weather(df)

    out_path = _finalize_and_save(df)
    return out_path


if __name__ == "__main__":
    from src.utils.cli import parse_force_flag
    from src.utils.logging_setup import get_logger
    get_logger()
    run(force=parse_force_flag())
