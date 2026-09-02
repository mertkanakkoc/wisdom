# wisdom — Status

A running log of what's built, why key decisions were made, and what's deliberately deferred. Kept in sync as phases complete.

## Faz 1 — `Column<T>` / `Table` core

`Table` (`src/models/table.rs`) is feature-complete for its original scope: `from_csv`, `file_path`, `row_count`, `column_count`, `element_count`, `get_column::<T>`, `update_column`, `add_column`, `remove_column`.

**Core design:**
- `Table` stores raw CSV data as `HashMap<String, ColumnData>`, where `ColumnData` is an enum (`Int(Column<i64>)`, `Float(Column<f64>)`, `Text(Column<String>)`, `Bool(Column<bool>)`, `Raw(Vec<Option<String>>)`). Columns start as `Raw` and get materialized into a typed `Column<T>` lazily, on first `get_column::<T>` call, then persisted back in place.
- **Checkout pattern** instead of `Clone`: `get_column` *moves* a `Column<T>` out of `Table` (via `HashMap::remove`), giving the caller true ownership — avoids needing `Column<T>: Clone`, which is blocked anyway since `behaviors` holds `Box<dyn Fn>` closures. A parallel `checkouts: HashMap<String, ColumnState>` (`Available`/`CheckedOut`) tracks checkout state; `get_column` refuses a second checkout until `update_column` writes it back.
- **Dispatch via `ColumnDataVariant` trait** (`wrap`/`unwrap`) resolves which `ColumnData` variant a generic `T` maps to, without runtime `TypeId`/`Any` checks.
- **Row-length invariant**: all columns must have equal length. `update_column`/`add_column` reject a mismatch against `row_count()`, with a `row_count() == 0` bypass for the "table is empty because its only column is checked out" edge case.
- **No dedicated rename**: done by the caller (`get_column` + `Column::update_name` + `add_column` under the new name) — avoids keeping `Table`'s `HashMap` key in sync with `Column`'s own `name` field.
- **`remove_column`** deletes regardless of checkout state.

**Deferred:** RAII `Drop`-based `ColumnGuard<T>` to auto-write-back a checked-out column instead of manual `update_column` — not needed for correctness, would just remove the "forgot to write back" risk.

## Faz 2 — missing values, scaling, train/test split

**Missing-value handling** (`src/models/column/fill.rs`): `missing_indices()`, `fill_with(value)` (generic, `T: Clone`), `fill_mean()` (`Column<f64>` only — averaging inherently produces a float), `forward_fill()`/`backward_fill()` (leading/trailing gaps with no valid neighbor stay `None`). All reuse `Column::update_element` for atomic writes.

**Normalization** (`src/models/column/scale.rs`): `min_max_scale()` and `z_score_standardization()` (population std dev), both `Column<f64>` only, both reuse `ColumnError::AllMissingElements`/`FilledElementsEqual`.

**Train/test split** (`src/models/table/split.rs`): `Table::train_test_split(&self, ratio, seed, columns: Option<Vec<String>>) -> (Table, Table)`. Deterministic-but-shuffled via `StdRng::seed_from_u64(seed)` + `SliceRandom::shuffle`. `columns: None` resolves to `self.checkouts.keys()`, not `self.data.keys()` (a checked-out column is absent from `data` but still tracked in `checkouts` — deriving "all columns" from `data` would silently drop it). Added `ColumnData::select_rows(&self, indices) -> ColumnData`, reused later by `FeatureStore::reconstruct_table`.

## Faz 3 — tensor adapter + training loop

**`Table::to_tensor`** (`src/models/table/tensor.rs`): `(feature_columns, target_column, device) -> Result<(Tensor, Tensor), TableError>`. `X` shape `(row_count, feature_columns.len())`, `y` shape `(row_count,)` (1D, matching `candle`'s own bias-vector convention). Row-major data built by hand rather than column-major + `.transpose()` (verified `transpose()` is a lazy, non-contiguous view via `candle`'s source). Only accepts columns already materialized to `Int`/`Float`/`Bool` — `Raw` is rejected as `NonNumericColumn` even if the strings would parse. Errors: `NonNumericColumn`, `MissingValuesPresent`, `TensorCreationFailed(candle_core::Error)`, `TargetColumnInFeatures` (catches target/data leakage).

**`train_linear_regression`** (`src/ml/training/linear_reg.rs`, moved here in Faz 5 — see below): `VarMap`/`VarBuilder` + `candle_nn::linear` + `AdamW`, full-batch gradient descent. Deliberately minimal scope — no reusable `Model` trait yet (only one concrete model exists; that abstraction waits for a second real case, per Faz 7). Test threshold (`final_loss < 5.0`) is a deliberately generous fixed bound, not a tight one — `VarBuilder`'s unseeded random init makes final loss vary a lot run to run; worth revisiting with real RNG seeding if this ever proves flaky.

## Faz 4 — `FeatureStore` (versioning + lineage + snapshots + drift) — complete

Lives in `src/feature_store.rs`, top-level (sibling to `models`/`ml`) — a capability built on top of the core data model, not part of it. Deliberately unified (versioning + lineage + snapshots + drift together), matching how real feature stores (Feast/MLflow) are structured, rather than four separate bolt-on features.

**Versioning & lineage:**
- Atomic unit = a single `Column` ("feature"), not a whole `Table`.
- Manual commit only (`FeatureStore::commit`) — `Table` stays the live/mutable working copy, `FeatureStore` only records history when explicitly told to. "Latest" always means "the last thing committed," not "what `Table` currently holds."
- Storage: full copy per version + a store-wide `max_versions` retention cap. After each `commit`, if a feature's version count exceeds `max_versions`, the oldest version *after* the permanently-kept first one is pruned (`versions.remove(1)`). `max_versions` must be `>= 2`.
- Lineage: `enum Transformation { FillWith, FillMean, ForwardFill, BackwardFill, MinMaxScale, ZScoreStandardization, Custom(String) }`, supplied explicitly by the caller at `commit` time — never inferred.
- Version numbers auto-increment per feature name, starting at 1 (matches MLflow's Model Registry convention).
- `get_version`/`get_latest_version` both borrow (`&ColumnVersion`), no ownership transfer needed.
- **Deferred:** delta-based storage / `Rc`/`Arc` structural sharing, as an optimization path if full-copy storage ever becomes a real cost. A future bulk-commit API would also need the pruning logic revisited (it currently assumes at most one excess version per `commit` call).

**Snapshots** (`src/feature_store/snapshot.rs`):
- `create_snapshot(&mut self, ordered_features: Vec<(String, usize)>, label: Option<String>) -> Result<SnapshotId, FeatureStoreError>` — pins a whole *combination* of feature versions as one named unit (the thing a model is actually trained/served against). Feature order is significant and preserved exactly as given — it's part of the computed `SnapshotId`, and later drives tensor-layout reconstruction without the caller having to remember the order themselves. Rejects duplicate feature names (`DuplicateFeatureInSnapshot`) and validates every `(name, version)` pair via `get_version` *before* storing anything (atomic — a failed call never touches `self.snapshots`).
- `SnapshotId(String)` — hex-encoded SHA-256 digest, computed via length-prefixing (`name.len()` as 8 bytes, then `name` bytes, then `version` as 8 bytes) rather than a separator byte, so there's no assumption needed about what bytes a column name can/can't contain.
- `get_snapshot(&self, id) -> Result<&Snapshot, FeatureStoreError>`.
- `reconstruct_table(&self, id) -> Result<Table, FeatureStoreError>` — rebuilds a brand-new, independent `Table` from a snapshot, walking `ordered_features` and reusing `ColumnData::select_rows` (passing all indices) to get an owned copy without needing `Column<T>: Clone`.

**Drift detection** (`src/feature_store/drift.rs`), deliberately built last:
- Atomic unit is a single feature — matches real tools' practice (Evidently/NannyML do per-column tests, with table-level rollups as a separate layer). **Deferred:** a `Table`/dataset-level drift summary that calls `detect_drift` per feature and aggregates results — not built, noted for later.
- `detect_drift(&self, name, baseline_version, current_version, threshold) -> Result<bool, FeatureStoreError>` — compares two versions of the same feature.
- Covers the same numeric types `to_tensor` accepts (`Int`/`Float`/`Bool`, cast to `f64`), for consistency with what's actually trainable. `Text`/`Raw` rejected via `NonNumericFeature`.
- Metric: normalized mean shift, `|current_mean - baseline_mean| / baseline_std`, falling back to the plain absolute difference when `baseline_std == 0` (avoids divide-by-zero). Returns a plain `bool`, not a report struct — simplest sufficient option.
- Missing values are skipped (not treated as `0`) when computing mean/std; a version that's *entirely* missing errors via `AllValuesMissing` rather than silently producing `NaN`.
- Added `Column<T>::present_values(&self) -> impl Iterator<Item = &T>` (public, in `column.rs`) — needed because drift's helper lives outside `models::column`'s module tree and couldn't reach the private `self.data.iter().flatten()` trick `fill_mean`/`z_score_standardization` already used internally. Retrofitted into those two plus `min_max_scale`, replacing their inline versions.
- Considered but rejected a shared `mean_and_std(values: &[f64])` helper — the calculation is short enough (~5 lines) that extracting it added more indirection than it saved. If a second real consumer (beyond `detect_drift`) ever wants it, the natural home would be a new top-level `src/stats.rs` module, sibling to `models`/`ml`/`feature_store`.

Branch `feature-snapshot` — merged. `ColumnVersion` and `Snapshot` later gained full public getter APIs (`ColumnVersion::{data, version_number, name, transformation, timestamp}`, `Snapshot::{id, ordered_features, timestamp, label}`) once Faz 5 needed to inspect them from outside `feature_store`.

## Faz 5 — embedded feature serving / inference — complete

Lives in `src/ml/`, split into two capabilities that deliberately avoid the word "serving" for anything Faz 5 builds (reserved for Faz 6's actual networked layer): **training** (`src/ml/training.rs` + `training/{linear_reg.rs, artifact.rs}`) and **inference** (`src/ml/inference.rs`, a single file). Both `training`/`artifact`/`linear_reg` are private submodules with `pub use` re-exports at the `ml::training` level — mirrors `feature_store`'s `snapshot`/`drift` pattern (which itself needed a `pub use` fix mid-Faz-5: its submodules were already private but had no re-export, silently making `SnapshotId`/`Snapshot` unreachable from outside `feature_store` until something outside it — `ml::training::artifact` — needed them for the first time).

**Training output & persistence:**
- `ModelArchitecture` enum (`Linear { in_features, out_features }` — one variant per known model type, same pattern as `Transformation`; a `Model` trait was considered and rejected again here, same reasoning as Faz 3).
- `TrainingOutput<M> { pub model: M, pub varmap: VarMap, pub architecture: ModelArchitecture }` — generic over `M` at near-zero cost (it's only ever *stored*, never called into, unlike a `Model` trait would require); fields are `pub` since there's no invariant to protect (unlike `ModelArtifact`). `train_linear_regression`'s return type changed from `candle_core::Result<Linear>` to `candle_core::Result<TrainingOutput<Linear>>`.
- `ModelArtifact { architecture, snapshot_id: SnapshotId, label: String, timestamp }` — private fields, `label` has a real invariant (non-empty, ≤ `MAX_LABEL_CHARACTER` = 64, allowlist-only `[a-zA-Z0-9_-]`) since it becomes part of a filename. `SnapshotId` gained a `Display` impl (raw hex, unlike `Debug`'s `SnapshotId("...")`) specifically to build that filename (`format!("{}_{}", snapshot_id, label)`).
- Persistence uses `candle_nn::VarMap::{save, load}` (safetensors format) — chosen over the VarMap-free alternative (`candle_core::safetensors` + `Linear::{weight, bias}`, confirmed viable but doesn't generalize past one `Linear` layer) specifically because a 2nd model type (Faz 7) is a real planned need. Metadata (architecture/snapshot_id/label/timestamp) is a JSON sidecar file next to the `.safetensors` weights (matches real practice — HF repos, MLflow), via `serde`/`serde_json` — `SystemTime` isn't natively `serde`-compatible, handled with a small hand-written `#[serde(with = "unix_seconds")]` module rather than a new dependency.
- `ModelArtifact::save(&self, varmap, dir) -> Result<String, ArtifactError>` writes weights *before* metadata (a crash mid-save leaves only a harmless orphan weights file; metadata's presence signals "complete"). `ModelArtifact::load(snapshot_id, label, dir, device) -> Result<(LoadedModel, ModelArtifact), ArtifactError>` is an associated function, not `&self` (no artifact exists yet at load time); rebuilds a same-shaped untrained model via `VarBuilder`/`candle_nn::linear` (using the *same* variable path, `"linear"`, as training) *before* calling `VarMap::load`, since `load` only fills already-declared variables. Returns a plain tuple, not a new struct (unlike `TrainingOutput`) — 2 self-disambiguating values didn't earn a new type the way `TrainingOutput`'s 3 semantically-distant ones did.
- `LoadedModel` enum (`Linear(Linear)`) — kept separate from `ModelArchitecture`: the latter is pure shape metadata serialized into JSON, so merging in real weights would mean trying (and failing — `Tensor` isn't `Serialize`) to write tensor data into the metadata file, duplicating what `.safetensors` already stores.
- A model saved on one device loads fine on another — `.safetensors` stores plain numeric data, not a device commitment (same as `torch.load(..., map_location=...)`).

**Inference** (`src/ml/inference.rs`):
- `Table::to_feature_tensor(feature_columns, device) -> Result<Tensor, TableError>` (in `models/table/tensor.rs`, alongside `to_tensor`) — a features-only tensor builder, since `to_tensor` requires a target column (it was built for training, where the true `y` must be known) and inference generally doesn't have one. Deliberately **not** sharing implementation with `to_tensor` — extracting a shared per-column-list helper would mean calling it twice (once for features, once for target) to avoid re-splitting a combined result, which means genuinely re-scanning the table's data twice; not worth it given `to_tensor` was already working and tested, so `to_feature_tensor` duplicates the loop independently instead.
- `run_inference_from_snapshot(loaded_model, feature_store, snapshot_id, device)` — batch/evaluation scenario: reconstructs the snapshot's `Table`, converts it via `to_feature_tensor` in the snapshot's own recorded order, runs the model.
- `run_inference_from_values(loaded_model, feature_names: &[String], values: &HashMap<String, f64>, device)` — live/new-data scenario, for a single observation. Values are looked up *by name*, not position, specifically to avoid the same "silent wrong order" risk `Snapshot` was designed to prevent; `feature_names` (typically fetched once from the artifact's snapshot, reused across many calls) fixes both required names and assembly order regardless of the `HashMap`'s own unordered iteration. Confirmed this is genuinely Faz 5's job, not Faz 6's — accepting live values is just a function parameter, no network involved; Faz 6 would only add a transport layer on top of this already-existing function.
- Both funnel into a small shared `run(loaded_model, x) -> Result<Tensor, InferenceError>` core (the one truly common step: `.forward()`, dispatched on `LoadedModel`'s variant).

**Deferred, real (not speculative) future work**: persisting/logging live inference inputs back into `FeatureStore`, both for future retraining and to make `detect_drift` genuinely useful over time (drift detection needs accumulating real production data to compare against a baseline). Not built now because it needs a foundational capability that doesn't exist yet — `Table` has no row-append operation (`add_column` only adds whole columns), and `FeatureStore`'s versioning is column-oriented/batch, not row-append-oriented.

## Not yet started

- **Faz 6** — low-latency online serving (a real paradigm shift: sync library → async server; meant to be revisited as its own deliberate decision, not assumed).
- **Faz 7** — learned index / self-profiling storage.

See the project roadmap for the full long-term vision.
