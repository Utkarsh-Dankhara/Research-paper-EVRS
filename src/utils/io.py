"""
src/utils/io.py

Small shared filesystem helpers used across every stage.
"""
import json
from pathlib import Path


def ensure_dirs(*paths):
    """Creates every directory in `paths` (and parents) if it doesn't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def ensure_project_dirs():
    """Creates every output directory the pipeline writes to. Centralized
    here instead of the original notebook's repeated per-cell mkdir loop."""
    import config
    ensure_dirs(*config.ALL_DIRS)


def save_json(path, obj):
    """Writes `obj` as pretty-printed JSON, creating parent dirs as needed.
    Non-JSON-native values (e.g. numpy floats, Path) are stringified rather
    than raising."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def output_exists(path) -> bool:
    """Used by every stage's skip-if-cached check."""
    return Path(path).exists()
