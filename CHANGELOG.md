# Changelog

All notable changes to `sheaf` (formerly `lean-prover`) are documented in this file.

## `[1.2.1]` - 2026-11-22 (`66b2487`, tag `v1.2.1`)

### Changed
- Emitted theorem-first `simp` and `split` candidates before blind definition unfolding in `synthesize_from_model_and_vdb`.
- Added split-aware and theorem-context-aware closers inside `run_best_first_tree_search` subgoal expansion (`crates/lean-prover/src/lib.rs`).
- Prioritized direct constructor `cases` + leaf `omega` ahead of double `fun_cases` fan-out for theorems with 3 or more structure binders.

### Fixed
- Added synthetic compound diamond-join regression test `test_diamond_compound_term_theorem_first_rewrite_and_split`.

## `[1.2.0]` - 2026-11-22 (`c7fa9c2`, tag `v1.2.0`)

### Added
- Topological import-DAG sorting (`topo_sort_workspace_files`) and transitive workspace symbol propagation (`ModuleExports`) in `discover_sorry_theorems`.
- Live in-memory `TreeSnapshot` upstream module composition (`compose_live_eval_source`) so dry-run verification never reads stale on-disk `.olean` files.
- Nested `match` `split` induction synthesis for custom recursive ASTs.

### Fixed
- Stripped trailing `/-- ... -/` docstrings and `@[...]` attributes before `set_option maxHeartbeats` in `verify_candidates_batched_with_heartbeats` (`crates/lean-prover/src/kernel_eval.rs`).
- Handled `sheaf prove --help` cleanly (exit code `0`) and rejected unknown `--*` flags with exit code `2`.

## `[1.1.0]` - 2026-11-22 (`71b30a8`, tag `v1.1.0`)

### Added
- Content-addressed in-memory VFS (`BlobHash`, `BlobStore`, `TreeSnapshot`) in `crates/lean-prover/src/vfs.rs` with atomic `--write` commits at exit.
- Deep Best-First compositional search tree (`crates/lean-prover/src/search.rs`) with two-tier deduplication (`(tree_hash, canon_subgoal)` before `Lean-Sys1`, `EpisodicCache` after).
- Hybrid kernel evaluator (`crates/lean-prover/src/kernel_eval.rs`) with per-probe `set_option maxHeartbeats` isolation and multi-file ephemeral `sheaf_wt_*` worktrees (`--max-workers`).
- CLI flags `--max-depth`, `--max-nodes`, `--max-workers`, and `--max-heartbeats`.

## `[1.0.0]` - 2026-11-21 (`3601432`, tag `v1.0.0`)

### Changed (Breaking)
- Renamed the CLI binary from `lean-prover` to `sheaf`.

## `[0.10.0]` - 2026-11-21 (`2366395`, tag `v0.10.0`)

### Added
- CLI flags `--gate-pct` and `--max-branches` (`--premise-tau` added in `6fcdb16`).

### Changed
- Executed the full RoPE attention, SwiGLU FFN, and masked attention pooling transformer graph in `LeanSys1OnnxSession::forward_with_gate` with `< 1e-4` ONNX Runtime parity (`crates/lean-sys1-ort/src/lib.rs`).
- Removed premature `PremiseFit` truncation to `2` entries, gated VDB queries on `FallbackToLeJepa`, and added direct SQ8 HNSW traversal (`search_sq8_hnsw`) without startup `f32` corpus dequantization (`crates/lean-vdb/src/lib.rs`).
- Enforced the strict 3-axiom allowlist (`propext`, `Classical.choice`, `Quot.sound`) and batched single-file kernel probing (`__probe_i`).

## `[0.9.1]` - 2026-11-21 (`a1f4a21`, tag `v0.9.1`)

### Fixed
- Replaced `try-rename_i` `List` induction fallback with explicit `nil`/`cons` nested case destructuring.
- Ordered bare `simp` and `omega` ahead of unrelated intra-module theorems and deduplicated synthesized candidates.

## `[0.9.0]` - 2026-11-21 (`3293d86`, tag `v0.9.0`)

### Added
- Balanced-binder theorem target parser, per-theorem file buffer isolation, `inductive`/`abbrev`/theorem discovery, and model-guided structural induction.

## `[0.8.0]` - 2026-11-21 (`1d0a712`, tag `v0.8.0`)

### Changed
- Removed fallback output stubs in favor of real embedded ONNX + LVDB inference and 5-stage live CLI telemetry.

## `[0.7.0]` - 2026-11-21 (`2a94950`, tag `v0.7.0`)

### Added
- Multi-line signature discovery, `lake env lean --json` resolution, and release diagnostics.

## `[0.6.0]` - 2026-11-21 (`1ac98ae`, tag `v0.6.0`)

### Added
- Embedded `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` directly into the Rust binary.

## `[0.5.0]` - 2026-11-21 (`413c2ed`, tag `v0.5.0`)

### Added
- Initial `lean-prover` CLI (`prove`, `check`, `init`, `doctor`), `orion` `v0.1.0` dependency pin, and `README.md`.

## `[0.4.0]` - 2026-11-21 (`3e1d712`, tag `v0.4.0`)

### Added
- Best-First search loop, 2-layer soft-token adapter, PDA identifier logit mask, prefix-replay module reload, and single-retry error repair.

## `[0.3.0]` - 2026-11-21 (`4a408d2`, tag `v0.3.0`)

### Added
- Pure-Rust calibrated `< 0.50` confidence gate, 4d `PremiseFit` scorer, and `FallbackOutcome` selector (`crates/lean-sys1-ort`).

## `[0.2.0]` - 2026-11-21 (`1c8479f`, tag `v0.2.0`)

### Added
- Offline `mathlib_sq8.vdb` SQ8 + HNSW index, `import_dag` transitive reduction, and `EpisodicCache` loop pruning (`crates/lean-vdb`).

## `[0.1.0]` - 2026-11-21 (`adc3c97`, tag `v0.1.0`)

### Added
- Lean 4 expression canonicalizer (`crates/lean-canonicalizer`), Pantograph JSON-RPC client and `#print axioms` gate (`crates/lean-repl`), and trace extraction scaffolding.
