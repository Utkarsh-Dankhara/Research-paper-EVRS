"""
src/data/loaders.py

Shared read helpers used by every downstream stage, so nothing has to
know the caching/file layout except this module:
  - load_city_graph / load_city_graphs   -> per-city OSM graphs (cached)
  - load_processed_data                   -> combined multi-city CSV
  - load_sequence_cache                    -> cached train/val/test splits + scalers
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def load_city_graph(city_code: str, place_name: str, force_refresh: bool = False):
    """Downloads (or loads from local cache) the drivable road network for
    one city. Boundary-based extraction (graph_from_place) is used so each
    city's full administrative extent is captured.

    Imports osmnx lazily -- this is the only function in the project that
    needs it, so preprocess.py/sequence_builder.py need it installed, but
    the 5 model scripts (which only call load_sequence_cache /
    load_processed_data below) don't."""
    import osmnx as ox

    cache_path = config.GRAPH_CACHE_DIR / f"{city_code}.graphml"

    if cache_path.exists() and not force_refresh:
        logger.info(f"[{city_code}] loading cached graph -> {cache_path}")
        return ox.load_graphml(cache_path)

    logger.info(f"[{city_code}] downloading '{place_name}' from OSM ...")
    G = ox.graph_from_place(place_name, network_type=config.NETWORK_TYPE)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, cache_path)
    logger.info(f"[{city_code}] cached -> {cache_path} | nodes={len(G.nodes)}, edges={len(G.edges)}")
    return G


def load_city_graphs(force_refresh: bool = False) -> dict:
    """Loads (or downloads) every city configured in config.CITIES.
    Returns {city_code: MultiDiGraph}."""
    return {
        city_code: load_city_graph(city_code, place_name, force_refresh=force_refresh)
        for city_code, place_name in config.CITIES.items()
    }


def load_processed_data() -> pd.DataFrame:
    """Loads the combined multi-city processed dataset produced by
    src/data/preprocess.py. Raises a clear error if that stage hasn't
    been run yet, instead of a confusing pandas FileNotFoundError."""
    if not config.PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"{config.PROCESSED_CSV} not found. Run `python main.py --stage preprocess` "
            f"(or `python -m src.data.preprocess`) first."
        )
    return pd.read_csv(config.PROCESSED_CSV)


def load_sequence_cache(include_full: bool = False) -> dict:
    """Loads the cached train/val/test sequence splits + fitted scalers
    built by src/data/sequence_builder.py. This is what every model
    module reads from, instead of each rebuilding sequences itself
    (rebuilding the predecessor adjacency is the expensive step this
    cache avoids repeating). Raises a clear error if that stage hasn't
    been run yet.

    y_train/y_val/y_test are still RobustScaler-scaled, same as the
    original notebook -- inverse-transform with y_scaler when you need
    real-scale values for metrics/plots.

    include_full=True also loads X_full/y_full (every row, unsplit,
    same order as row_meta.csv) -- only needed for the Bayesian model's
    full-graph inference step that builds the routing map.
    """
    npz_path = config.SEQUENCES_DIR / "sequences.npz"
    meta_path = config.SEQUENCES_DIR / "sequence_meta.json"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run `python main.py --stage sequences` "
            f"(or `python -m src.data.sequence_builder`) first."
        )

    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)

    out = {
        "X_train": data["X_train"], "X_val": data["X_val"], "X_test": data["X_test"],
        "y_train": data["y_train"], "y_val": data["y_val"], "y_test": data["y_test"],
        "x_scaler": joblib.load(config.SCALERS_DIR / "x_scaler.joblib"),
        "y_scaler": joblib.load(config.SCALERS_DIR / "y_scaler.joblib"),
        "feature_cols": meta["feature_cols"],
    }
    if include_full:
        out["X_full"] = data["X_full"]
        out["y_full"] = data["y_full"]
        out["row_meta"] = pd.read_csv(config.SEQUENCES_DIR / "row_meta.csv")
    return out
