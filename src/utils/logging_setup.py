"""
src/utils/logging_setup.py

Configures the shared project logger (console + a per-run file under
logs/run_<timestamp>.log) and tracks per-stage/per-model status so
main.py can print a run summary even when some stages fail.
"""
import logging
import sys
from datetime import datetime

import config

_LOGGER_NAME = "evrs"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Configures (once) the ROOT logger's level + handlers, then returns
    a logger with the given name.

    FIX: this used to configure only the logger literally named "evrs".
    Every other module in the project gets its logger the standard way --
    logging.getLogger(__name__), e.g. "src.models.random_forest" -- which
    is NOT a child of "evrs" in the logger hierarchy (hierarchy is by
    dotted name; "src.models.random_forest"'s ancestors are "src.models"
    -> "src" -> root, never "evrs"). With only "evrs" configured, root's
    default level (WARNING) silently swallowed every logger.info() call
    everywhere else in src/ -- they were never even created as records,
    so nothing was actually broken or skipped, it just never printed.
    Only logger.warning()/.exception() calls leaked through, via Python's
    "last resort" stderr fallback for unconfigured loggers at WARNING+.
    That's why e.g. random_forest.run()'s "Skipping training..." /
    "Training Random Forest (500 trees)..." lines (both .info()) never
    appeared even when that code definitely ran, while
    bayesian_bilstm.run()'s missing-routing-map message (a .warning())
    did.

    Configuring root fixes every module's logger.info() call at once,
    since all of them propagate to root by default -- no per-file changes
    needed. Third-party libraries that also log via the stdlib (e.g.
    matplotlib, urllib3) may now also emit INFO-level lines that were
    previously silent; if that gets noisy, set individual loggers back to
    WARNING with logging.getLogger("matplotlib").setLevel(logging.WARNING)
    etc.
    """
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S")

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        root.addHandler(console)

        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.LOGS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        logging.getLogger(_LOGGER_NAME).info(f"Logging to {log_path}")

    return logging.getLogger(name)


class RunSummary:
    """Tracks per-stage/per-model status (ok / failed / skipped) across a
    full main.py run, so a final summary can be printed even when some
    stages fail. Usage:

        summary = RunSummary()
        try:
            ...
            summary.add("bayesian_bilstm", "ok", "MAE=0.0421")
        except Exception as e:
            summary.add("bayesian_bilstm", "failed", str(e))
        ...
        summary.print_summary(logger)
    """

    _ICONS = {"ok": "OK", "failed": "FAILED", "skipped": "SKIPPED"}

    def __init__(self):
        self._entries = []

    def add(self, name: str, status: str, detail: str = ""):
        self._entries.append((name, status, detail))

    def any_failed(self) -> bool:
        return any(status == "failed" for _, status, _ in self._entries)

    def print_summary(self, logger=None):
        log = logger.info if logger else print
        log("=" * 60)
        log("RUN SUMMARY")
        log("=" * 60)
        for name, status, detail in self._entries:
            tag = self._ICONS.get(status, status.upper())
            suffix = f"  {detail}" if detail else ""
            log(f"[{tag:<7}] {name:<20}{suffix}")
        log("=" * 60)