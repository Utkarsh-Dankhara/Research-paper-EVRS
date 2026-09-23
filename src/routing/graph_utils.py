"""
src/routing/graph_utils.py

Ported from Phase 4 setup of notebooks/EVRS_NN_multi_models.ipynb:
  - make_haversine(G) -> A* heuristic bound to one city's graph
  - compute_city_centers(graphs) -> centroid of each city's own nodes
  - random_od_pair(G, city_code, ...) -> a routable, sufficiently-far-
    apart random origin/destination pair
  - show_pick_map(city_code, city_centers) -> click-to-read-coordinates
    map, used by the dashboard's manual OD-selection mode

Used by both the routing pipeline stage (src.routing.router,
src.routing.bootstrap) and dashboard/app.py.
"""
import logging
import math

import folium
import networkx as nx
import numpy as np

import config

logger = logging.getLogger(__name__)


def make_haversine(G):
    """Returns a haversine heuristic (metres) bound to one city's graph, for A*."""
    def haversine(node1, node2):
        n1, n2 = G.nodes[node1], G.nodes[node2]
        lat1, lon1 = math.radians(n1["y"]), math.radians(n1["x"])
        lat2, lon2 = math.radians(n2["y"]), math.radians(n2["x"])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6_371_000 * 2 * math.asin(math.sqrt(a))
    return haversine


def compute_city_centers(graphs: dict) -> dict:
    """Centroid of each city's own graph nodes -- works for any city in
    config.CITIES without hardcoding coordinates."""
    return {
        code: (
            float(np.mean([d["y"] for _, d in G.nodes(data=True)])),
            float(np.mean([d["x"] for _, d in G.nodes(data=True)])),
        )
        for code, G in graphs.items()
    }


def _largest_component_nodes(G):
    """A*/routing only works within one connected component -- sample from the largest."""
    Gu = G.to_undirected()
    largest_cc = max(nx.connected_components(Gu), key=len)
    return list(largest_cc)


def random_od_pair(G, city_code: str, min_distance_m: float = None, max_tries: int = None, rng=None):
    """
    Randomly picks a start/end node pair inside one city's graph,
    guaranteed to be in the same connected component (so a route always
    exists) and at least min_distance_m apart (so it doesn't pick two
    nodes on the same tiny street). Returns
    (start_lat, start_lon, end_lat, end_lon, label).
    """
    min_distance_m = config.MIN_OD_DISTANCE_M if min_distance_m is None else min_distance_m
    max_tries = config.MAX_OD_TRIES if max_tries is None else max_tries
    rng = rng or np.random.default_rng()

    haversine = make_haversine(G)
    nodes = _largest_component_nodes(G)

    n1 = n2 = None
    for _ in range(max_tries):
        n1, n2 = rng.choice(nodes, size=2, replace=False)
        if haversine(n1, n2) >= min_distance_m:
            lat1, lon1 = G.nodes[n1]["y"], G.nodes[n1]["x"]
            lat2, lon2 = G.nodes[n2]["y"], G.nodes[n2]["x"]
            label = f"Random pair in {city_code.upper()} ({haversine(n1, n2) / 1000:.1f} km apart)"
            return lat1, lon1, lat2, lon2, label

    # fallback: take the last sampled pair even if under the distance threshold
    lat1, lon1 = G.nodes[n1]["y"], G.nodes[n1]["x"]
    lat2, lon2 = G.nodes[n2]["y"], G.nodes[n2]["x"]
    return lat1, lon1, lat2, lon2, f"Random pair in {city_code.upper()} (below distance threshold)"


def show_pick_map(city_code: str, city_centers: dict):
    """
    Click-to-read-coordinates folium map for manual origin/destination
    selection. Click anywhere -- a popup shows that point's lat/lon.
    Used by the dashboard's manual OD-selection mode (and standalone, if
    someone wants to pick coordinates outside the dashboard too).
    """
    lat, lon = city_centers[city_code]
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="cartodbpositron")
    m.add_child(folium.LatLngPopup())
    return m
