"""
src/data/sequence_builder.py

Builds the SEQ_LEN-step predecessor-chain sequences used by every model,
fits the feature/target RobustScalers, and produces the train/val/test
split. Ported from Phase 2 of notebooks/EVRS_NN_multi_models.ipynb,
including the ~300x perf fix (vectorized numpy-array lookups instead of
df.iloc calls in the hot loop).

Runs ONCE and caches its output to data/processed/sequences/ +
models/scalers/, so all 5 model scripts read from the same cache instead
of each rebuilding the (expensive) predecessor adjacency themselves --
this is the main runtime-reduction lever for repeat training runs.

Also caches the full (unsplit) X_full/y_full + row_meta.csv, needed by
the Bayesian model's full-graph inference step (see
src/models/bayesian_bilstm.py) to build the routing map -- not needed by
the other 4 models.

Public entry point: run(force: bool = False) -> dict
"""
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import config
from src.data.loaders import load_processed_data, load_city_graphs, load_sequence_cache
from src.utils.io import ensure_project_dirs, save_json, output_exists

logger = logging.getLogger(__name__)


def _build_highway_ohe(df: "pd.DataFrame") -> "np.ndarray":
    """
    One-hot encode the highway column into config.N_HIGHWAY_CLASSES binary columns.

    FIX: road_risk/lane_risk/speed_risk were highway-type lookup scalars —
    keeping them meant the model learned to reconstruct the label formula
    rather than discovering genuine risk. OHE gives the raw type as binary
    indicators; the model learns its own relationship between type and risk.

    WHY appended AFTER scaling: binary {0,1} columns must NOT be passed
    through RobustScaler (it would compress them toward zero). Numeric
    features are scaled first; OHE is concatenated to the right.
    Final shape per timestep: 13 numeric (scaled) + 9 OHE = 22 features.
    """
    import numpy as np
    hw = df["highway"].fillna("other").astype(str)
    ohe = np.zeros((len(df), config.N_HIGHWAY_CLASSES), dtype=np.float32)
    for i, cls in enumerate(config.HIGHWAY_CLASSES):
        ohe[:, i] = (hw == cls).astype(np.float32)
    # Unknown highway types → "other" column (last index)
    ohe[~hw.isin(config.HIGHWAY_CLASSES).values, -1] = 1.0
    return ohe


def _fit_scalers_and_features(df):
    """
    Fits RobustScaler on 13 numeric CANDIDATE_FEATURE_COLS, then appends
    the 9-class highway OHE block (unscaled) to produce X_all of shape
    (N, 22) — the fixed feature dimensionality every model expects.

    FIX: road_risk, lane_risk, speed_risk, access_risk removed from features.
    These were lookup-table encodings of highway type — keeping them caused
    the model to trivially reconstruct the label (GBM R²=0.99 on real data).
    Highway information is preserved via OHE; the model discovers its own
    weights for each road type from the risk_score_v2 distribution.
    """
    import numpy as np
    feature_cols = [c for c in config.CANDIDATE_FEATURE_COLS if c in df.columns]
    missing = [c for c in config.CANDIDATE_FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"CANDIDATE_FEATURE_COLS missing from dataset: {missing}")

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_numeric = x_scaler.fit_transform(df[feature_cols].fillna(0).values)  # (N, 13)
    X_ohe     = _build_highway_ohe(df)                                      # (N, 9)
    X_all     = np.concatenate([X_numeric, X_ohe], axis=1)                  # (N, 22)

    y_all = y_scaler.fit_transform(df[[config.TARGET_COL]].values)

    config.SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, config.SCALERS_DIR / "x_scaler.joblib")
    joblib.dump(y_scaler, config.SCALERS_DIR / "y_scaler.joblib")

    # FIX: feature_cols used to be returned as just the 13 numeric names,
    # but X_all is already 22-wide (13 numeric + 9 highway OHE) by this
    # point. Every model reads len(feature_cols) back out of
    # sequence_meta.json to build its input layer / flatten its feature
    # names, so an under-counted list here silently desyncs from the real
    # data width -- that's what produced the (5,13) vs (5,22) Keras shape
    # error and the tree-model feature-importance IndexError. Appending
    # the OHE names keeps feature_cols an accurate 1:1 description of
    # X_all's last dimension.
    highway_ohe_names = [f"highway_{c}" for c in config.HIGHWAY_CLASSES]
    all_feature_names = feature_cols + highway_ohe_names

    total = len(all_feature_names)
    logger.info(
        f"Features: {len(feature_cols)} numeric (scaled) + {config.N_HIGHWAY_CLASSES} "
        f"highway OHE = {total} total per timestep | Target: {config.TARGET_COL}"
    )
    return X_all, y_all, all_feature_names, x_scaler, y_scaler


def _build_predecessor_adjacency(df, graphs):
    """Strips the per-city osmid namespace back off (raw OSM way ids are
    globally unique, so merging the per-city adjacency dicts into one
    combined lookup keyed on the raw id is safe), then builds one merged
    predecessor adjacency dict across all cities' graphs."""
    df = df.copy()
    df["osmid_local"] = df.apply(lambda r: str(r["osmid"])[len(str(r["city_code"])) + 1:], axis=1)

    osmid_to_rows = {}
    for idx, row in df.iterrows():
        osmid_to_rows.setdefault(row["osmid_local"], []).append(idx)

    edge_adjacency = {}
    for city_code, G_city in graphs.items():
        for u, v, key, data in G_city.edges(keys=True, data=True):
            eid = data.get("osmid", "")
            eid = str(eid[0]) if isinstance(eid, list) else str(eid)
            for pu, pv, pk, pdata in G_city.in_edges(u, keys=True, data=True):
                pid = pdata.get("osmid", "")
                pid = str(pid[0]) if isinstance(pid, list) else str(pid)
                if pid != eid:
                    edge_adjacency.setdefault(eid, []).append(pid)

    osmid_to_first_row = {k: v[0] for k, v in osmid_to_rows.items()}
    return df, edge_adjacency, osmid_to_first_row


def _build_sequences(df, X_all, y_all):
    """Vectorized predecessor-chain sequence construction (SEQ_LEN steps
    back per road, longest-segment tiebreak among multiple predecessors,
    left-padded by repeating the first element when a chain runs out of
    predecessors before SEQ_LEN). Pulls columns to plain numpy arrays once
    and indexes those in the hot loop -- ~300x faster than a naive
    df.iloc-per-step version at multi-city scale (measured: ~24s for 8k
    rows -> would project to ~5 min for 100k+ rows with df.iloc)."""
    graphs = load_city_graphs()  # cached by now (preprocess already ran), so this is fast
    df, edge_adjacency, osmid_to_first_row = _build_predecessor_adjacency(df, graphs)

    osmid_local_arr = df["osmid_local"].to_numpy()
    length_arr = df["length"].to_numpy()

    def get_sequence_indices(target_idx, n_steps):
        chain = [target_idx]
        current = osmid_local_arr[target_idx]
        for _ in range(n_steps - 1):
            preds = edge_adjacency.get(current, [])
            if not preds:
                break
            best, best_len = None, -1.0
            for p in preds:
                r = osmid_to_first_row.get(p)
                if r is not None and length_arr[r] > best_len:
                    best_len, best = length_arr[r], r
            if best is None:
                break
            chain.append(best)
            current = osmid_local_arr[best]
        chain.reverse()
        while len(chain) < n_steps:
            chain.insert(0, chain[0])
        return chain[:n_steps]

    X_seqs = np.array(
        [X_all[get_sequence_indices(i, config.SEQ_LEN)] for i in range(len(df))],
        dtype=np.float32,
    )
    y_seqs = y_all[:, 0].astype(np.float32)
    logger.info(f"Sequence dataset: X={X_seqs.shape}, y={y_seqs.shape}")
    return df, X_seqs, y_seqs


def run(force: bool = False) -> dict:
    """
    Builds the SEQ_LEN-step predecessor sequences, fits the RobustScalers,
    and produces the train/val/test split -- runs ONCE and caches
    everything, so all 5 model modules read from the same cache instead
    of each rebuilding this.

    Skips entirely if the cache already exists and force=False.
    Returns the same dict shape as src.data.loaders.load_sequence_cache().
    """
    npz_path = config.SEQUENCES_DIR / "sequences.npz"
    x_scaler_path = config.SCALERS_DIR / "x_scaler.joblib"
    y_scaler_path = config.SCALERS_DIR / "y_scaler.joblib"
    cache_complete = output_exists(npz_path) and output_exists(x_scaler_path) and output_exists(y_scaler_path)

    if cache_complete and not force:
        logger.info(f"Skipping sequence building -- {npz_path} already exists "
                    f"(pass force=True to rerun).")
        return load_sequence_cache()
    if output_exists(npz_path) and not cache_complete:
        logger.warning(f"{npz_path} exists but its scalers are missing (models/scalers/) -- "
                        f"cache is inconsistent, rebuilding from scratch instead of crashing later.")

    ensure_project_dirs()

    df = load_processed_data()
    logger.info(f"Loaded {len(df):,} road segments across {df['city_code'].nunique()} cities "
                f"| Target: {config.TARGET_COL}")

    X_all, y_all, feature_cols, x_scaler, y_scaler = _fit_scalers_and_features(df)
    df, X_seqs, y_seqs = _build_sequences(df, X_all, y_all)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_seqs, y_seqs, test_size=config.TEST_SIZE, random_state=config.SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=config.VAL_SIZE_OF_REMAINDER, random_state=config.SEED)

    logger.info(f"Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    config.SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        X_full=X_seqs, y_full=y_seqs,
    )
    save_json(config.SEQUENCES_DIR / "sequence_meta.json", {"feature_cols": feature_cols})
    df[["osmid", "osmid_local", "city_code", "city", "highway", "travel_time"]].to_csv(
        config.SEQUENCES_DIR / "row_meta.csv", index=False
    )
    logger.info(f"Cached sequences -> {npz_path}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "x_scaler": x_scaler, "y_scaler": y_scaler, "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    from src.utils.cli import parse_force_flag
    from src.utils.logging_setup import get_logger
    get_logger()
    run(force=parse_force_flag())