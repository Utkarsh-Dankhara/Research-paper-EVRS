"""
main.py -- single entry point for the whole EVRS pipeline.

Stages (in order): preprocess -> sequences -> train (5 models, isolated)
-> compare -> route -> (optional) launch dashboard.

Usage:
    python main.py                                   # full pipeline, skip-if-cached
    python main.py --force                            # ignore caches, rerun everything
    python main.py --stage train --models rf,transformer
    python main.py --stage route
    python main.py --stage compare
    python main.py --launch-dashboard

Every stage is wrapped in try/except and recorded in a RunSummary
printed at the end. preprocess/sequences are true hard dependencies (
nothing else can run without them) so a failure there stops the run;
everything from `train` onward degrades gracefully -- one model or the
routing stage failing does not stop the others.
"""
import argparse
import sys
import time

import config
from src.utils.io import ensure_project_dirs
from src.utils.logging_setup import RunSummary, get_logger

STAGE_CHOICES = ["all", "preprocess", "sequences", "train", "compare", "route"]

_MODEL_ALIASES = {
    "rf": "random_forest", "random_forest": "random_forest",
    "xgb": "xgboost", "xgboost": "xgboost",
    "lstm": "vanilla_lstm", "vanilla_lstm": "vanilla_lstm",
    "transformer": "transformer",
    "bilstm": "bayesian_bilstm", "bayesian_bilstm": "bayesian_bilstm",
}


def _banner(logger, text):
    """Prints a visually distinct block so it's always obvious what's
    currently running, both in the console and in the persistent log
    file under logs/."""
    logger.info("")
    logger.info("#" * 64)
    logger.info(f"#  {text}")
    logger.info("#" * 64)


def _parse_args():
    p = argparse.ArgumentParser(description="EVRS pipeline orchestrator")
    p.add_argument("--stage", default="all", choices=STAGE_CHOICES,
                    help="Which stage to run (default: all)")
    p.add_argument("--models", default=None,
                    help="Comma-separated subset of models to train, e.g. 'rf,transformer'. "
                         "Accepts short names (rf,xgb,lstm,transformer,bilstm) or full module "
                         "names. Default: all 5.")
    p.add_argument("--force", action="store_true",
                    help="Ignore every cache and rerun every selected stage from scratch.")
    p.add_argument("--launch-dashboard", action="store_true",
                    help="After the run finishes, launch the Streamlit routing dashboard.")
    return p.parse_args()


def _resolve_models(models_arg):
    if not models_arg:
        return list(config.MODEL_NAMES)
    resolved = set()
    for tok in models_arg.split(","):
        tok = tok.strip().lower()
        if tok not in _MODEL_ALIASES:
            valid = sorted(set(_MODEL_ALIASES))
            raise SystemExit(f"Unknown model '{tok}'. Choices: {valid}")
        resolved.add(_MODEL_ALIASES[tok])
    # keep config.MODEL_NAMES' canonical order rather than the user's typed order
    return [m for m in config.MODEL_NAMES if m in resolved]


def _run_preprocess(force, summary, logger):
    _banner(logger, "STAGE: preprocess (multi-city OSM extraction + feature engineering)")
    t0 = time.time()
    from src.data import preprocess
    try:
        preprocess.run(force=force)
        summary.add("preprocess", "ok", f"{(time.time() - t0) / 60:.1f} min")
    except Exception:
        logger.exception("preprocess failed")
        summary.add("preprocess", "failed")
        summary.print_summary(logger)
        raise SystemExit(
            "\nPreprocessing failed -- this is a hard dependency for every later stage, so "
            "stopping here. See the traceback above (and logs/) for details."
        )


def _run_sequences(force, summary, logger):
    _banner(logger, "STAGE: sequences (predecessor-chain sequence building + scaling)")
    t0 = time.time()
    from src.data import sequence_builder
    try:
        sequence_builder.run(force=force)
        summary.add("sequences", "ok", f"{(time.time() - t0) / 60:.1f} min")
    except Exception:
        logger.exception("sequence building failed")
        summary.add("sequences", "failed")
        summary.print_summary(logger)
        raise SystemExit(
            "\nSequence building failed -- this is a hard dependency for every model, so "
            "stopping here. See the traceback above (and logs/) for details."
        )


def _run_train(models, force, summary, logger):
    from src.models import bayesian_bilstm, random_forest, transformer, vanilla_lstm, xgboost_model
    from src.models.base import run_model_stage

    module_map = {
        "bayesian_bilstm": bayesian_bilstm,
        "random_forest": random_forest,
        "xgboost": xgboost_model,
        "vanilla_lstm": vanilla_lstm,
        "transformer": transformer,
    }
    for name in models:
        _banner(logger, f"TRAINING MODEL: {name}  ({models.index(name) + 1}/{len(models)})")
        t0 = time.time()
        run_model_stage(name, module_map[name].run, summary=summary, force=force)
        logger.info(f"--- {name} stage finished ({(time.time() - t0) / 60:.1f} min) ---")


def _run_compare(summary, logger):
    _banner(logger, "STAGE: compare (cross-model comparison)")
    from src.evaluation import compare
    try:
        result = compare.run()
        n = len(result.get("available_models", []))
        summary.add("compare", "ok", f"{n}/{len(config.MODEL_NAMES)} models included")
    except Exception:
        logger.exception("comparison failed")
        summary.add("compare", "failed")


def _run_route(force, summary, logger):
    """Routing has two independent sub-stages, each wrapped separately so
    one failing doesn't block the other:
      - router.run(): one representative route per city -> folium maps +
        reliability profile charts (what the dashboard also draws on).
      - bootstrap.run(): config.N_PAIRS_PER_CITY OD pairs per city,
        routed and bootstrap-resampled into the statistical risk-
        reduction claim (route_comparison_metrics.csv, bootstrap_ci.csv).
    Both depend on data/routing/bayesian_routing_map.csv, produced by
    training the Bayesian BiLSTM -- if that hasn't run yet, both fail
    with a clear message rather than crashing the pipeline."""
    _banner(logger, "STAGE: route (maps + reliability + bootstrap validation)")
    try:
        from src.routing import router
        router.run(force=force)
        summary.add("route_maps", "ok")
    except Exception as e:
        logger.warning(f"Route map generation skipped: {e}")
        summary.add("route_maps", "skipped", str(e))

    try:
        from src.routing import bootstrap
        result = bootstrap.run(force=force)
        detail = ""
        if result.get("mean_reduction_pct") is not None:
            sig = "significant" if result.get("significant") else "not significant"
            detail = f"mean risk reduction {result['mean_reduction_pct']:.1f}% ({sig})"
        summary.add("route_bootstrap", "ok", detail)
    except Exception as e:
        logger.warning(f"Bootstrap validation skipped: {e}")
        summary.add("route_bootstrap", "skipped", str(e))


def main():
    args = _parse_args()
    logger = get_logger()
    ensure_project_dirs()
    summary = RunSummary()
    run_t0 = time.time()

    models = _resolve_models(args.models)
    stages = (["preprocess", "sequences", "train", "compare", "route"]
              if args.stage == "all" else [args.stage])

    if "preprocess" in stages:
        _run_preprocess(args.force, summary, logger)
    if "sequences" in stages:
        _run_sequences(args.force, summary, logger)
    if "train" in stages:
        _run_train(models, args.force, summary, logger)
    if "compare" in stages:
        _run_compare(summary, logger)
    if "route" in stages:
        _run_route(args.force, summary, logger)

    logger.info(f"\nTotal run time: {(time.time() - run_t0) / 60:.1f} min")
    summary.print_summary(logger)

    if args.launch_dashboard:
        import subprocess
        logger.info("Launching dashboard: streamlit run dashboard/app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])

    if summary.any_failed():
        sys.exit(1)


if __name__ == "__main__":
    main()
