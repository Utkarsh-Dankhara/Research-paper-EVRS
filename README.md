# EVRS — Emergency Vehicle Routing System

Restructured from two monolithic notebooks (`dataset_full_preprocess.ipynb`,
`EVRS_NN_multi_models.ipynb`) into an isolated, script-driven pipeline.
Single entry point: `python main.py`.

## Status

Fully implemented, tested, and audited. Every stage was actually executed
end-to-end against real (synthetic-graph) data, including the full
preprocessing pipeline itself — not just downstream stages against
pre-made fixtures. A second full audit pass re-read every file, checked
cross-file consistency, and live-tested every fix, catching several real
bugs along the way (see below). The final `python main.py` run with
production hyperparameters completed cleanly:
`preprocess → sequences → all 5 models → comparison → route maps → bootstrap validation`.

**Training target: `risk_score_v2`, not `risk_score_v1`.** `risk_score_v1`
(the original hand-designed formula) is still computed and kept in the
processed CSV for reference, but it's built almost entirely out of the
same columns the models take as input features — a model doesn't need to
learn anything to reconstruct it, it just needs to do arithmetic on its
own inputs (verified: a zero-ML formula reconstruction using only the
model's own features hits R²=0.99998, actually *beating* the trained
model). `risk_score_v2` (`src/data/risk_v2.py`) is the real target now —
built from per-city rank/percentile normalization of structurally-
grounded factors, clipped to `[0,1]`, with no direct 1:1 overlap with the
raw feature columns.

## ✅ Resolved — BiLSTM sigmoid/scaling bug (was: R²=0.60, now: R²=0.9929)

`models/bayesian_bilstm/model.keras` previously had a bug: its final
layer used `activation="sigmoid"` while being trained against
`RobustScaler`-transformed targets, which aren't bounded to `[0,1]`
(roughly half land as negative numbers a sigmoid can never reach).
Result: predictions collapsed to two flat plateaus, R²=0.60, vs.
0.92–0.99 for the other 4 models on the identical target.

Fixed by switching to `activation="linear"` (matching
`vanilla_lstm.py`/`transformer.py`, which use the same target scaling
correctly) and retraining. Confirmed in `results/bayesian_bilstm/metrics.json`:
**MAE=0.0046, RMSE=0.0097, R²=0.9929** — now the best of all 5 models on
every metric (previously XGBoost led; see the note on `comparison_summary.json`
staleness below). The routing map, bootstrap results, and route maps
were regenerated from the corrected model and reflect this fix.

If you retrain the BiLSTM again in the future for any reason, rerun
`python main.py --stage compare` afterward too --
`results/comparison/comparison_summary.json` only updates when that
stage runs, so it can silently go stale relative to a fresher
`metrics.json` the way it briefly did here.

RF, XGBoost, Vanilla LSTM, and Transformer were never affected by this
bug — it was specific to the BiLSTM's output layer.

## Session fixes (2026-09) — feature-count mismatch, early stopping, mixed precision, logging, sigmoid/scaling bug

- **`sequence_meta.json` under-reported the feature count**: after adding
  9 one-hot highway columns (13 numeric + 9 OHE = 22 features/timestep),
  `sequence_builder.py` still saved only the 13 numeric names to
  `feature_cols`. Every Keras model builds its input layer from
  `len(feature_cols)`, so all three (`bayesian_bilstm`, `vanilla_lstm`,
  `transformer`) crashed with `expected shape=(None,5,13), found
  shape=(None,5,22)`. Same root cause also crashed `random_forest`'s and
  `xgboost`'s feature-importance plot (`IndexError: list index out of
  range` — 110 actual importances vs. 65 names). Fixed by returning the
  full 22-name list (13 numeric + `highway_<class>` × 9) from
  `_fit_scalers_and_features()`. **Requires** `--stage sequences --force`
  to regenerate the stale `sequence_meta.json` — retraining alone won't
  pick this up.
- **`EarlyStopping` never actually triggered**: default `min_delta=0`
  means *any* improvement, even in the 6th decimal place, resets the
  patience counter — so training could run to the full `EPOCHS` ceiling
  even when `val_loss` looked completely flat for 12+ epochs. Added
  `EARLY_STOP_MIN_DELTA = 1e-4` (`config.py`), passed into `EarlyStopping`
  in `src/models/base.py`.
- **Mixed-precision reload crash**: `bayesian_bilstm.py`'s "model already
  trained, just regenerate the routing map" shortcut path called
  `load_model()` without first re-enabling the `mixed_float16` global
  policy. Standard layers survive this fine (their dtype is saved in
  their own config), but `TemporalAttention`'s `self.W`/`self.V` Dense
  sublayers are freshly constructed in `__init__` and pick up whatever
  the *global* policy is at that exact moment — so they came back as
  float32 while the surrounding BiLSTM output was float16, crashing with
  `Input 'y' ... has type float16 that does not match type float32`.
  Fixed by calling `maybe_enable_mixed_precision()` at the top of
  `load_trained_model()` in all three Keras model modules.
- **Module-level loggers were silently dropped**: `logging_setup.get_logger()`
  only configured a logger literally named `"evrs"`. Every other module's
  `logging.getLogger(__name__)` (e.g. `"src.models.random_forest"`) is
  *not* a child of `"evrs"` in the logger hierarchy, and with no handler
  anywhere in its own ancestry, Python's default root level (`WARNING`)
  silently dropped every `.info()` call — only `.warning()`/`.exception()`
  calls leaked through via Python's stderr "last resort" fallback. This
  is why e.g. `random_forest.run()`'s "Skipping training..." /
  "Training Random Forest (500 trees)..." lines never appeared even when
  that code definitely ran. Fixed by configuring the root logger instead
  of just `"evrs"` — every module's logger propagates to root by default,
  so this fixes every existing `.info()` call at once, no per-file changes
  needed.
- **Sigmoid output vs. `RobustScaler`-scaled target mismatch** — see the
  pending-retrain note above.

- **Real data-quality bug**: `_clean_lanes()` relied on `float(val)` raising
  for missing lane counts to trigger its highway-type fallback estimate —
  but `float(nan)` does not raise, so missing lane data silently became
  `NaN` (later `0` after `fillna(0)`) instead of a sensible estimate,
  corrupting a real training feature. Fixed and verified by re-running
  preprocessing and confirming zero unexpected NaNs.
- `main.py` was silently missing the route-map generation stage (only
  ran the statistical bootstrap, never the actual map/reliability charts).
- Cache-consistency gaps in `sequence_builder.py` and `router.py` (checked
  one file's existence but not its paired dependency, risking a raw
  crash instead of a clear message on a partially-cleared cache).
- `bayesian_bilstm.py` couldn't recover if `data/routing/` was cleared
  separately from the trained model — now detects this and regenerates
  just the routing map, verified live.
- None of the "standalone" module scripts actually parsed `--force` from
  the command line despite being documented that way — fixed with a
  shared `src/utils/cli.py` helper across all 9 relevant modules.
- `compare.py` would crash entirely if one model's `metrics.json` were
  corrupted, instead of excluding just that model — fixed and verified
  with a deliberately corrupted fixture.
- `bootstrap.py`'s statistical conclusion (mean risk reduction, 95% CI,
  significance) was only logged, never saved — now persisted to
  `bootstrap_summary.json`.
- A self-caught mistake: an attempted cosmetic fix to `TemporalAttention`
  briefly broke model reloading (list/tuple concatenation error) — caught
  by re-testing the reload path immediately after, not assumed safe.

## Reviewer-response tooling

Built to directly answer specific review comments with real evidence
rather than just prose:

- **`--mode reviewer`** (`python -m src.models.ablation --mode reviewer`)
  -- ablates exactly the 3 model-architecture components a review named
  by name (Bidirectional, Attention, MC Dropout). Temperature Scaling
  is answered automatically (every MC-dropout variant now reports
  calibration before/after in its `metrics.json`, needing no separate
  trained model); Bayesian Routing Cost Function is answered by the
  existing routing/bootstrap pipeline, not model training -- see the
  module docstring in `src/models/ablation.py` for the full mapping.
- **`src/evaluation/uncertainty_analysis.py`** -- builds a reliability
  diagram (nominal vs. observed coverage, before/after temperature
  scaling) and an uncertainty-vs-error plot (does actual error increase
  in the model's own high-uncertainty predictions? -- the quantified
  answer to "why prefer this model over one with higher point accuracy
  but no uncertainty signal"). Reads what `bayesian_bilstm.py` already
  saved, no retraining needed: `python -m src.evaluation.uncertainty_analysis`.
- **Routing performance timing** -- `src/routing/bootstrap.py` now times
  every `run_route()` call and saves `data/routing/routing_performance.json`
  (mean/median/p95/max ms to compute all 5 route modes for one OD pair),
  with an explicit note distinguishing this from one-time model training
  cost -- the real number for a "real-time deployment feasibility"
  question.

## EDA notebook

`notebooks/eda_data_exploration.ipynb` -- viewer-only exploratory analysis
of the preprocessed multi-city dataset (no training, no reprocessing).
Requires `python main.py --stage preprocess` to have run at least once.
Covers: dataset overview, per-city summary, missing/estimated-tag
sparsity, feature distributions, correlation with `risk_score_v1`, an
overview map with all 3 cities at once, static per-city risk snapshots,
and an interactive per-city risk map (segment-sampled above 8,000 rows
for rendering performance).

## Ablation study

`src/models/ablation.py` isolates the 5 architectural decisions stacked
inside the Bayesian BiLSTM (bidirectional, stacked depth, attention,
predecessor-sequence context, MC-Dropout/Bayesian uncertainty), holding
everything else fixed. This is different from the 5-model comparison
above, which compares different model families.

```bash
python -m src.models.ablation --mode leave_one_out    # 7 variants: full model, each component removed once, minimal baseline
python -m src.models.ablation --mode incremental       # 6 variants: minimal baseline building up to the full model
python -m src.models.ablation --mode full_factorial    # 32 variants: every combination (generated, not hand-listed) -- slow
```

Not wired into `main.py --stage all` — run it deliberately when you want
it, not on every pipeline run. Saves to `models/ablation/<variant>/` and
`results/ablation/<variant>/`, plus a combined bar chart + CSV summary
(with % MAE delta vs. the full model) in `results/ablation/comparison/`.

## Recent fixes (training speed, visibility, retraining, map UI)

- **Speed**: the Bayesian BiLSTM's uncertainty estimation was calling
  Keras's `.predict()` in a loop 60 times, which rebuilds its internal
  data pipeline on every call and was directly causing `tf.function`
  retracing (confirmed by promoting the warning to a hard error and
  re-testing) — switched to direct model calls, `BATCH_SIZE` raised to
  256, and an opt-in `ENABLE_MIXED_PRECISION` flag added for Tensor-Core
  GPUs (RTX 20xx+/30xx).
- **A real epochs/early-stopping bug**: `EPOCHS=10` with
  `early_stop_patience=20` meant early stopping could never trigger —
  every run always trained the fixed 10-epoch ceiling regardless of
  convergence. Fixed to `EPOCHS=60` / `EARLY_STOP_PATIENCE=12` /
  `PLATEAU_PATIENCE=5`, so training now genuinely stops on its own once
  validation loss plateaus.
- **Visibility**: added clear stage/model banners with per-stage timing
  to `main.py`, and fixed a progress-logging condition (`% 25`) that
  could never fire for `NUM_PASSES=20`, silencing the single longest
  phase of training entirely. Same fix applied to preprocessing's
  betweenness-centrality step, which had no progress indication either.
- **Retraining**: already solved by skip-if-cached (checks
  `models/<name>/model.keras` before training) — use `--force` to
  override, e.g. `python main.py --stage train --models bilstm --force`.
- **Map UI**: `folium.LayerControl(collapsed=False, ...)` was forcing
  the route/legend selector to render permanently expanded, blocking
  the map. Changed to `collapsed=True` (click-to-expand icon instead).

## Scope

Multi-city: Gandhinagar, Ahmedabad, Surat (`gnr` / `ahd` / `srt`), all
extracted fresh from OSM. The old Gandhinagar-only `data/*.csv` and trained
model artifacts from the original project have been moved to
`extras/legacy_gnr_only_data/` and `extras/legacy_gnr_only_models/` — kept
for reference, not used by the new pipeline.

## First run

```bash
pip install tensorflow xgboost streamlit streamlit-folium
python main.py
```

The first run downloads all 3 cities from OSM (slow, especially
Ahmedabad/Surat) and trains all 5 models with production settings
(EPOCHS=200 ceiling with early stopping, NUM_PASSES=100 MC-Dropout
passes). Every stage after that is skip-if-cached, so repeat runs are
fast unless you pass `--force`.

## Layout

```
main.py                 single entry point / orchestrator
config.py                all shared settings (cities, hyperparams, paths)
requirements.txt

data/
  raw/graph_cache/        per-city .graphml, cached from OSM on first run
  processed/               roads_all_cities_processed.csv
    sequences/              cached X/y sequence splits + scalers (built once, shared by all models)
  routing/                 bayesian_routing_map.csv, route comparison metrics, bootstrap CI

models/                   trained artifacts, one subfolder per model + shared scalers/
results/                  metrics + predictions, one subfolder per model, plus comparison/
routes/                   folium maps (maps/) and reliability charts (reliability/)

src/
  data/                    preprocessing + sequence building + loaders
  models/                  5 independent, self-contained model modules + base.py interface
  evaluation/              shared metrics/plots + cross-model comparison
  routing/                 A* routing, OD pair generation, reliability, bootstrap CI
  utils/                   logging + io helpers

dashboard/app.py          Streamlit app — interactive route/reliability viewer
notebooks/                 optional, viewer-only (no training/preprocessing code allowed here)
extras/                    archived material (old routes, backup notebooks, legacy gnr-only data/models, paper .tex files)
logs/                      per-run logs
```

## Design principles

- **No shared in-memory state across stages.** Every stage reads/writes
  files. This is what makes each model trainable independently.
- **Every model script is self-contained and independently runnable**
  (e.g. `python -m src.models.transformer`) — it loads cached sequence
  data itself and writes its own results, without depending on any other
  model's run. Verified by deliberately crashing one model mid-run and
  confirming the others still trained and saved their results.
- **`main.py` wraps every stage in try/except.** One model failing is
  logged and skipped; the rest of the pipeline (and the comparison stage)
  still completes. preprocess/sequences are the only hard dependencies
  (nothing can run without them); everything from `train` onward degrades
  gracefully.
- **Skip-if-cached at every stage** — OSM downloads, graph adjacency
  building, and sequence/scaler construction only happen once unless
  `--force` is passed. Cache consistency is checked (not just file
  existence) before trusting a skip.
- **Results are separate AND combined.** Each model writes its own
  `results/<model>/metrics.json`; `src/evaluation/compare.py` reads
  whichever of the 5 exist and builds the combined comparison — it
  degrades gracefully if a model is missing.
- **Routing has two independent sub-stages**: `router.run()` (one
  representative route per city → folium maps + reliability charts, also
  what the dashboard draws on) and `bootstrap.run()` (many OD pairs →
  statistical risk-reduction claim with a bootstrap 95% CI). Each fails
  independently.

## Usage

```bash
pip install tensorflow xgboost streamlit streamlit-folium

python main.py                                   # full pipeline, skip-if-cached
python main.py --force                            # ignore caches, rerun everything
python main.py --stage train --models rf,transformer
python main.py --stage route
python main.py --stage compare

streamlit run dashboard/app.py                    # interactive route viewer
```