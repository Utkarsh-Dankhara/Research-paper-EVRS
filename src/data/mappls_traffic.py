"""
src/data/mappls_traffic.py
----------------------------
Fetches live-traffic-adjusted travel times for primary/secondary road
segments via the Mappls Distance Matrix API, and computes a
live_congestion_factor = traffic_duration / baseline_duration per
segment (>1.0 = slower than free-flow, ~1.0 = no congestion).

NOT wired into main.py --stage all -- this is deliberate: it spends
real API credits, so it only ever runs when you explicitly call it.

SAFETY:
- --dry-run makes ZERO network calls. Always run this first.
- --limit caps how many segments get fetched in one run (default 20).
- Results are written to the cache file incrementally (after every
  segment), so if you Ctrl+C partway through -- e.g. to go check actual
  credit consumption in your Mappls console -- nothing already fetched
  is lost, and the next run picks up from there.
- Results already in the cache within MAPPLS_TRAFFIC_CACHE_TTL_HOURS
  (config.py) are reused, not re-fetched, unless --force-refetch is passed.

Two Mappls calls per segment: one to `distance_matrix` (no traffic,
baseline duration) and one to `distance_matrix_eta` (traffic delay
applied to the default route's duration). Both take the segment's own
start/end coordinates as a 2-point matrix -- this is the most
cost-predictable shape (exactly 2 calls per segment, easy to budget for)
even though it doesn't exploit the API's up-to-100-point batching. If you
find batching is meaningfully cheaper per segment after checking your
console, this is the place to change that -- MAPPLS_DISTANCE_MATRIX_MAX_POINTS
is already in config.py for exactly that purpose.

CONFIRMED (2026-09): this project's Mappls allocation includes "Distance
Matrix API Non Traffic" and "Distance Matrix ETA API Traffic" but NOT the
full re-routing "Distance Matrix API Traffic" -- that one 401s with
"Api Access Denied" regardless of key/app setup, because it's simply not
part of this app's allocated 20 APIs. distance_matrix_eta IS allocated
and returns real traffic-delay data (confirmed: 168.9s vs 147.2s baseline
for the same 720m segment, a legitimate ~14.7% congestion reading), so
that's the resource this script actually calls.

I could not find a reliable, current published price-per-call for this
API. Do NOT assume a cost -- run --dry-run, then a --limit 1 real test,
then check your actual Mappls console balance before and after to get
your real per-segment cost, before scaling up.

Usage:
    # ALWAYS first -- shows what would be fetched, spends nothing
    python -m src.data.mappls_traffic --city gnr --dry-run

    # smallest possible real test -- confirms the API call format is
    # correct and lets you check exact credit consumption in your console
    python -m src.data.mappls_traffic --city gnr --limit 1

    # small batch once the above looks right
    python -m src.data.mappls_traffic --city gnr --limit 20

    # scale up once you know your real per-segment cost
    python -m src.data.mappls_traffic --city gnr --limit 300
    python -m src.data.mappls_traffic --city all --limit 300

Output: data/raw/mappls_traffic_cache/{city_code}.json, keyed by
osmid_local. NOT merged into the processed CSV or risk_score_v2
automatically -- that's a deliberate follow-up step once you're satisfied
with coverage and cost.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config
from src.data.loaders import load_city_graph
from src.utils.mappls_auth import AuthError, get_rest_api_key

try:
    from dotenv import load_dotenv
    load_dotenv(config.BASE_DIR / ".env")
except ImportError:
    pass  # optional -- falls back to whatever's already exported in the shell

logger = logging.getLogger(__name__)

_DM_BASE = "https://route.mappls.com/route/dm"
_CACHE_DIR = config.RAW_DIR / "mappls_traffic_cache"
_REQUEST_TIMEOUT = 15
_REQUEST_PAUSE_SECONDS = 0.2  # be polite to the API; also naturally throttles spend


def _cache_path(city_code: str) -> Path:
    return _CACHE_DIR / f"{city_code}.json"


def _load_cache(city_code: str) -> dict:
    path = _cache_path(city_code)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Cache file {path} unreadable ({exc}); starting fresh.")
        return {}


def _save_cache(city_code: str, cache: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(city_code).write_text(json.dumps(cache, indent=2))


def _is_fresh(entry: dict) -> bool:
    fetched_at = datetime.fromisoformat(entry["fetched_at"])
    age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
    return age_hours < config.MAPPLS_TRAFFIC_CACHE_TTL_HOURS


def _normalize_highway(value) -> str:
    """Same convention as preprocess.py: OSM sometimes tags a list; take the first."""
    if isinstance(value, list):
        return value[0] if value else "unclassified"
    return value or "unclassified"


def _candidate_segments(city_code: str, place_name: str) -> list[dict]:
    """Every primary/secondary edge in the city's cached graph, with the
    coordinates and osmid_local needed to fetch + later merge traffic data."""
    G = load_city_graph(city_code, place_name)
    segments = []
    for u, v, key, data in G.edges(keys=True, data=True):
        highway = _normalize_highway(data.get("highway"))
        if highway not in config.MAPPLS_TRAFFIC_HIGHWAY_TYPES:
            continue
        osmid = data.get("osmid")
        if isinstance(osmid, list):  # OSM occasionally tags multiple way ids per edge
            osmid = osmid[0]
        u_lat, u_lon = G.nodes[u]["y"], G.nodes[u]["x"]
        v_lat, v_lon = G.nodes[v]["y"], G.nodes[v]["x"]
        segments.append({
            "osmid_local": str(osmid),
            "highway": highway,
            "u_lat": u_lat, "u_lon": u_lon,
            "v_lat": v_lat, "v_lon": v_lon,
        })
    return segments


def _fetch_duration(resource: str, seg: dict, api_key: str) -> float:
    """One Distance Matrix call for a single 2-point (start->end) segment.
    Returns duration in seconds. Raises requests.RequestException or
    ValueError on any failure -- caller decides how to handle it."""
    coords = f"{seg['u_lon']},{seg['u_lat']};{seg['v_lon']},{seg['v_lat']}"
    url = f"{_DM_BASE}/{resource}/driving/{coords}"
    params = {"region": "ind", "access_token": api_key}
    resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return float(data["results"]["durations"][0][1])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected response shape: {json.dumps(data)[:300]}") from exc


def run(city_code: str, place_name: str, limit: int, dry_run: bool, force_refetch: bool) -> None:
    logger.info(f"[{city_code}] Scanning cached graph for {config.MAPPLS_TRAFFIC_HIGHWAY_TYPES} segments...")
    segments = _candidate_segments(city_code, place_name)
    logger.info(f"[{city_code}] {len(segments)} candidate segments found.")

    cache = _load_cache(city_code)
    to_fetch = []
    for seg in segments:
        entry = cache.get(seg["osmid_local"])
        if entry is not None and not force_refetch and _is_fresh(entry):
            continue
        to_fetch.append(seg)
        if len(to_fetch) >= limit:
            break

    already_cached = len(segments) - len(to_fetch)
    print(f"\n[{city_code}] {len(segments)} candidate segments total")
    print(f"[{city_code}] {already_cached} already cached and fresh (within {config.MAPPLS_TRAFFIC_CACHE_TTL_HOURS}h)")
    print(f"[{city_code}] {len(to_fetch)} would be fetched this run (limit={limit})")
    print(f"[{city_code}] That's {len(to_fetch) * 2} API calls (2 per segment: baseline + traffic)\n")

    if dry_run:
        print("DRY RUN -- no network calls made. Remove --dry-run to actually fetch.")
        return

    if not to_fetch:
        print("Nothing to fetch (either no candidates, or everything's already cached and fresh).")
        return

    try:
        api_key = get_rest_api_key()
    except AuthError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("Fetching -- Ctrl+C at any point is safe, already-fetched segments are saved.\n")
    ok, failed = 0, 0
    for i, seg in enumerate(to_fetch, 1):
        try:
            baseline_s = _fetch_duration("distance_matrix", seg, api_key)
            time.sleep(_REQUEST_PAUSE_SECONDS)
            traffic_s = _fetch_duration("distance_matrix_eta", seg, api_key)
            time.sleep(_REQUEST_PAUSE_SECONDS)

            congestion = traffic_s / baseline_s if baseline_s > 0 else 1.0
            cache[seg["osmid_local"]] = {
                "highway": seg["highway"],
                "baseline_duration_s": baseline_s,
                "traffic_duration_s": traffic_s,
                "live_congestion_factor": congestion,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            _save_cache(city_code, cache)  # incremental -- survives an early stop
            ok += 1
            print(f"  [{i}/{len(to_fetch)}] osmid_local={seg['osmid_local']} "
                  f"congestion={congestion:.2f} ({ok} ok, {failed} failed)")
        except (requests.RequestException, ValueError) as exc:
            failed += 1
            logger.warning(f"[{city_code}] osmid_local={seg['osmid_local']} failed: {exc}")

    print(f"\n[{city_code}] Done. {ok} fetched, {failed} failed. Cache: {_cache_path(city_code)}")
    print("Check your Mappls console now for actual credits consumed by this run --")
    print(f"that's your real per-segment cost ({ok * 2} calls were made).")
    print("\nThis data is cached but NOT yet merged into the processed CSV or")
    print("risk_score_v2 -- that's a deliberate separate step once you're happy")
    print("with coverage and cost.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=list(config.CITIES.keys()) + ["all"], required=True)
    parser.add_argument("--limit", type=int, default=20,
                         help="Max segments to fetch this run (default: 20, deliberately conservative)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Show what would be fetched, make zero network calls")
    parser.add_argument("--force-refetch", action="store_true",
                         help="Ignore cache freshness, refetch everything up to --limit")
    args = parser.parse_args()

    cities = list(config.CITIES.items()) if args.city == "all" else [(args.city, config.CITIES[args.city])]
    for city_code, place_name in cities:
        run(city_code, place_name, args.limit, args.dry_run, args.force_refetch)


if __name__ == "__main__":
    main()