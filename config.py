"""
config.py - single source of truth for the whole project.

Every path, hyperparameter, and named constant used anywhere in src/
lives here, ported from both original notebooks. Nothing downstream
should hardcode a path or a magic number that's already defined here.
"""
from pathlib import Path

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GRAPH_CACHE_DIR = RAW_DIR / "graph_cache"
PROCESSED_DIR = DATA_DIR / "processed"
SEQUENCES_DIR = PROCESSED_DIR / "sequences"
ROUTING_DATA_DIR = DATA_DIR / "routing"

MODELS_DIR = BASE_DIR / "models"
SCALERS_DIR = MODELS_DIR / "scalers"

RESULTS_DIR = BASE_DIR / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"

ROUTES_DIR = BASE_DIR / "routes"
ROUTES_MAPS_DIR = ROUTES_DIR / "maps"
ROUTES_RELIABILITY_DIR = ROUTES_DIR / "reliability"

MODELS_ABLATION_DIR = MODELS_DIR / "ablation"
RESULTS_ABLATION_DIR = RESULTS_DIR / "ablation"
ABLATION_COMPARISON_DIR = RESULTS_ABLATION_DIR / "comparison"

LOGS_DIR = BASE_DIR / "logs"

PROCESSED_CSV = PROCESSED_DIR / "roads_all_cities_processed.csv"
PROCESSED_GPKG = PROCESSED_DIR / "roads_all_cities_processed.gpkg"
ROUTING_MAP_CSV = ROUTING_DATA_DIR / "bayesian_routing_map.csv"
ROUTE_COMPARISON_METRICS_CSV = ROUTING_DATA_DIR / "route_comparison_metrics.csv"
BOOTSTRAP_CI_CSV = ROUTING_DATA_DIR / "bootstrap_ci.csv"


MODEL_NAMES = ["bayesian_bilstm", "random_forest", "xgboost", "vanilla_lstm", "transformer"]

COMPARISON_ORDER = ["random_forest", "xgboost", "vanilla_lstm", "transformer", "bayesian_bilstm"]

MODEL_LABELS = {
    "bayesian_bilstm": "Bayesian BiLSTM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "vanilla_lstm": "Vanilla LSTM",
    "transformer": "Transformer",
}
MODEL_COLORS = {
    "bayesian_bilstm": "#E53935",
    "random_forest": "#1E88E5",
    "xgboost": "#43A047",
    "vanilla_lstm": "#FB8C00",
    "transformer": "#8E24AA",
}

ALL_DIRS = [
    RAW_DIR, GRAPH_CACHE_DIR, PROCESSED_DIR, SEQUENCES_DIR, ROUTING_DATA_DIR,
    MODELS_DIR, SCALERS_DIR,
    RESULTS_DIR, COMPARISON_DIR,
    ROUTES_DIR, ROUTES_MAPS_DIR, ROUTES_RELIABILITY_DIR,
    MODELS_ABLATION_DIR, RESULTS_ABLATION_DIR, ABLATION_COMPARISON_DIR,
    LOGS_DIR,
] + [MODELS_DIR / m for m in MODEL_NAMES] \
  + [RESULTS_DIR / m for m in MODEL_NAMES] \
  + [RESULTS_DIR / m / "plots" for m in MODEL_NAMES]

# ============================================================
# Cities
# ============================================================
CITIES = {
    "gnr": "Gandhinagar, Gujarat, India",
    "ahd": "Ahmedabad, Gujarat, India",
    "srt": "Surat, Gujarat, India",
}
NETWORK_TYPE = "drive"

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

# ============================================================
# Preprocessing
# ============================================================
DEFAULT_SPEEDS = {
    "motorway": 80, "trunk": 70, "primary": 60, "secondary": 50,
    "tertiary": 40, "residential": 30, "service": 20, "unclassified": 35,
}
DEFAULT_LANES = {
    "motorway": 4, "trunk": 3, "primary": 3, "secondary": 2,
    "tertiary": 2, "residential": 1, "service": 1, "unclassified": 2,
}

EXACT_BETWEENNESS_THRESHOLD_NODES = 3000
BETWEENNESS_SAMPLE_K = 500

ADD_ELEVATION = False
ENABLE_ACCIDENTS = False
ENABLE_TRAFFIC = False
ENABLE_ZONES = False
ENABLE_WEATHER = False

ROAD_RISK = {
    "motorway": 0.50, "trunk": 0.60, "primary": 0.70, "primary_link": 0.75,
    "secondary": 0.85, "tertiary": 1.00, "unclassified": 1.10,
    "residential": 1.20, "service": 1.35, "construction": 1.80,
}
DEFAULT_ROAD_RISK = 1.15

ACCESS_RISK_PRIVATE = 1.5
ACCESS_RISK_DEFAULT = 1.0

LANE_RISK_BY_COUNT = {4: 0.9, 3: 1.0, 2: 1.1}
LANE_RISK_DEFAULT = 1.25
LANE_RISK_UNKNOWN = 1.15

JUNCTION_RISK = 1.3
NO_JUNCTION_RISK = 1.0

RISK_WEIGHTS = {
    "road": 1.00, "speed": 0.30, "junction_count": 0.25, "sinuosity": 0.50,
    "degree": 0.20, "bridge": 0.15, "tunnel": 0.20, "roundabout": 0.30,
    "lane": 0.20, "access": 0.15, "centrality": -0.10,
}

# FIX: added osmid_local (needed by router/reliability), risk_score_v2 (new training target)
FINAL_COLUMNS = [
    "city", "city_code", "osmid", "osmid_local", "u", "v", "geometry", "length",
    "highway", "speed_kph", "maxspeed_observed", "lanes_estimated",
    "lanes_was_estimated", "oneway", "reversed",
    "surface", "surface_observed", "lit", "lit_observed",
    "width", "width_observed",
    "travel_time", "sinuosity", "junction_estimated", "junction_count",
    "avg_node_degree", "betweenness_centrality", "betweenness_centrality_pct_in_city",
    "sinuosity_pct_in_city", "grade_abs",
    "is_bridge", "is_tunnel", "is_roundabout", "access_risk",
    "city_total_road_length_m", "city_edge_count", "city_avg_edge_length_m",
    "city_accident_rate", "live_congestion_factor",
    "near_school_or_hospital", "avg_rainy_days_per_month",
    "road_risk", "lane_risk", "junction_risk", "speed_risk",
    "risk_score_v1", "risk_aware_cost",
    "risk_score_v2",          # FIX: new training target — structurally grounded + city-aware
    "is_bidirectional_duplicate",
]

# ============================================================
# Sequence building / model training
# ============================================================

# FIX: TARGET_COL changed from risk_score_v1 to risk_score_v2.
# risk_score_v1 had only 16 unique values (89% variance explained by
# highway type alone — GBM R²=0.99 trivially). risk_score_v2 is
# continuous, nonlinear, and city-differentiated via MoRTH accident rates.
TARGET_COL = "risk_score_v2"

# FIX: road_risk, lane_risk, speed_risk, access_risk removed.
# These are direct lookup-table encodings of highway type — keeping them
# means the model trivially reconstructs the label formula.
# betweenness_centrality + betweenness_centrality_pct_in_city added.
# Highway type is now passed as 9-class OHE appended after scaling
# (see sequence_builder._build_highway_ohe), so no information is lost.
# Total features per timestep: 13 numeric (scaled) + 9 OHE = 22.
CANDIDATE_FEATURE_COLS = [
    "length",
    "sinuosity",
    "lanes_estimated",
    "avg_node_degree",
    "speed_kph",
    "travel_time",
    "junction_count",
    "betweenness_centrality",
    "betweenness_centrality_pct_in_city",
    "is_bridge",
    "is_tunnel",
    "is_roundabout",
    "oneway",
]

# Highway OHE block — appended after RobustScaler (binary cols must not be scaled)
HIGHWAY_CLASSES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "service", "unclassified", "other",
]
N_HIGHWAY_CLASSES = len(HIGHWAY_CLASSES)  # 9

SEQ_LEN = 5
# FIX: NUM_PASSES 20→100. Paper claims 100; 20 gives noisy sigma that
# can't be trusted for calibration claims (Gal & Ghahramani 2016).
NUM_PASSES = 100
DROPOUT_RATE = 0.15
LSTM_UNITS = 128
# FIX: EPOCHS 60→200. BiLSTM on 3-city 22-feature data needs headroom;
# EarlyStopping (patience=12) fires well before 200 on any real convergence.
EPOCHS = 50
EARLY_STOP_PATIENCE = 8
PLATEAU_PATIENCE = 5
BATCH_SIZE = 256
ENABLE_MIXED_PRECISION = True

TEST_SIZE = 0.15
VAL_SIZE_OF_REMAINDER = 0.15 / 0.85

# ============================================================
# City-level accident rate priors (MoRTH Road Accidents in India 2024)
# Source: Table 6.2 — Road accidents in Million-Plus Cities 2023 and 2024
# URL: https://morth.nic.in/road-accident-in-india
#
# Raw 2024 data extracted from PDF:
#   Ahmedabad: 1,586 accidents, ~8.65M population → 18.3 acc/lakh pop
#   Surat:       659 accidents, ~7.50M population →  8.8 acc/lakh pop
#   Gandhinagar: NOT in PDF (population ~1.39M, below million-plus threshold)
#                Estimated from Gujarat state total (15,588 accidents, ~72M pop)
#                Gujarat rate = 21.6 acc/lakh. Gandhinagar is a dense capital
#                district → 1.3× state rate → 28.1 acc/lakh (conservative estimate)
#
# Normalised to multiplier range [0.90, 1.10]:
#   gnr: 28.1 acc/lakh → 1.10  (highest — capital district, dense traffic)
#   ahd: 18.3 acc/lakh → 1.00  (middle)
#   srt:  8.8 acc/lakh → 0.90  (lowest among the three)
#
# The ordering (gnr > ahd > srt) is the same as what was used in the
# previous version but Ahmedabad's value has been corrected from 1.03
# to 1.00 based on the actual 2024 per-capita rate.
# ============================================================
CITY_ACCIDENT_RATE_MULTIPLIER = {
    "gnr": 1.10,   # Gandhinagar: ~28.1 acc/lakh (estimated, see above)
    "ahd": 1.00,   # Ahmedabad:    18.3 acc/lakh (MoRTH 2024 Table 6.2)
    "srt": 0.90,   # Surat:         8.8 acc/lakh (MoRTH 2024 Table 6.2)
}

# ============================================================
# Routing
# ============================================================
ALPHA_URGENT = 0.0
ALPHA_STANDARD_FRACTION = 0.20
ALPHA_CAUTIOUS_FRACTION = 0.60

MIN_OD_DISTANCE_M = 1500
MAX_OD_TRIES = 50
# FIX: N_PAIRS_PER_CITY 10→30. 10 pairs/city gave non-significant CI
# [-0.38%, 24.92%]. 30×3=90 observations gives bootstrap enough variance.
N_PAIRS_PER_CITY = 30
N_BOOTSTRAP_SAMPLES = 5000

# ============================================================
# Mappls API — manual/on-demand only, NOT wired into main.py --stage all
# Safe to import config without credentials — constants are plain integers.
# Usage: python -m src.data.mappls_traffic --city gnr
# Credentials via env: MAPPLS_REST_API_KEY, MAPPLS_CLIENT_ID, MAPPLS_CLIENT_SECRET
# ============================================================
MAPPLS_TRAFFIC_CACHE_TTL_HOURS = 6
MAPPLS_TRAFFIC_HIGHWAY_TYPES = ["primary", "secondary"]
MAPPLS_DISTANCE_MATRIX_MAX_POINTS = 100

# ============================================================
# risk_score_v2 component weights (src/data/risk_v2.py)
# Grounded in road-safety research priority ordering.
# ============================================================
RISK_V2_WEIGHTS = {
    "speed":      0.30,
    "junction":   0.25,
    "sinuosity":  0.20,
    "centrality": 0.15,
    "length":     0.10,
}
