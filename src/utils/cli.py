"""
src/utils/cli.py

Tiny shared CLI parser for every module's `if __name__ == "__main__":`
block, so standalone runs (`python -m src.models.transformer --force`,
`python -m src.data.preprocess --force`, etc.) actually support --force
instead of always defaulting to force=False regardless of what's typed
on the command line.
"""
import argparse


def parse_force_flag() -> bool:
    """Parses just --force from sys.argv. Used by every module's
    standalone entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Ignore any existing cached output and rerun from scratch.")
    return parser.parse_args().force
