# sheaf

`sheaf` is a local-first compositional Lean 4 auto-prover built in Rust. It keeps memory and VRAM usage inside an 8 GB RAM / 4 GB VRAM envelope by combining a full RoPE transformer `Lean-Sys1` encoder, direct SQ8 HNSW premise retrieval, a content-addressed `BlobStore` VFS, Best-First deep tree search, and a real Lean 4 kernel verifier.

The compiled `sheaf` binary carries its own embedded `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` assets so it runs out of the box with zero external downloads or Python dependencies.

## Architecture Highlights (`v0.13.0`)

- **Content-Addressed VFS (`BlobStore` + `TreeSnapshot`):** Forks workspace states in O(1) via SHA-256 hashes and `Arc<str>` sharing so dead-end branches roll back cleanly and `--write` commits dirty files atomically at exit.
- **Deep Best-First Search & Two-Tier Dedup:** Expands subgoals up to `--max-depth` (`4`) and `--max-nodes` (`64`), short-circuiting exact `(tree_hash, canon_subgoal)` duplicates before `Lean-Sys1` and semantic loops (`cos > 0.98`) in `EpisodicCache` after `Lean-Sys1`.
- **Hybrid Kernel Verifier & Strict Axiom Gate:** Batches single-file probes with per-probe `set_option maxHeartbeats` isolation, evaluates multi-file branches in ephemeral `sheaf_wt_*` worktrees with early cancellation (`--max-workers`), and accepts proofs only under `propext`, `Classical.choice`, and `Quot.sound`.

## Quick Self-Verification Demo (`examples/sheaf_self_verify`)

The repository ships a standalone `import Init` specification workspace at `examples/sheaf_self_verify/` modeling `AxiomGate`, `GateSpec`, and `VfsSpec` invariants with `8` open `sorry` theorems:

```powershell
# 1. List the 8 open specification invariants (exits 1)
./target/release/sheaf check examples/sheaf_self_verify

# 2. Prove all 8 invariants cleanly against Lean 4 (exits 0)
./target/release/sheaf prove examples/sheaf_self_verify

# 3. Optional strict-lint in-place write mode (enforces -DwarningAsError=true)
./target/release/sheaf prove examples/sheaf_self_verify --write --strict-lint
```

## Operational Output Modes

- **Interactive TTY (default):** Renders a 12 Hz `indicatif` live branch viewport on `stderr` and collapses each verified theorem into a single permanent scrollback line.
- **Plain / CI (`--plain` or non-TTY `stderr`):** Disables spinners and ANSI escapes automatically so CI logs stay clean.
- **Verbose (`-v` / `--verbose`):** Prints the full 5-stage telemetry tree (`Canonicalize`, `Tokenize`, `Lean-Sys1`, `Orion VDB`, `Lean Kernel`) per theorem.
- **Machine JSON (`--json`):** Silences `stderr` and emits newline-delimited JSON (`NDJSON`) events to `stdout` for editor integrations and benchmark scripts.

## Empirical Benchmark Summary (`v1.8.0`)

| Suite | Scope | Proved | Notes |
| :--- | :--- | :---: | :--- |
| Single-module (`Test A1..A5`, `Test B1..B10`) | Arithmetic, `List`, `BinTree`, 3-record associativity | `15 / 15` | Zero `sorryAx` |
| 4-folder diamond (`Test C1..C5`) | Cross-module `import` composition | `5 / 5` | Per-occurrence `unusedSimpArgs` pruning on `--write` |
| 4-folder state-machine (`Test D1..D6`) | `structure` + `List` replay + dependent `∃` | `6 / 6` | Dynamic `DirectExecute` to `FallbackToLeJepa` escalation |
| Self-verification (`examples/sheaf_self_verify`) | `AxiomGate`, `GateSpec`, `VfsSpec` | `8 / 8` | Standalone `import Init` spec |

## Prerequisites

- **Lean 4.36.0** (`lean` and `lake` on `PATH` via `elan`)
- **Rust** (stable toolchain with `cargo`)

## How to Use

### 1. Build the Release Binary

```powershell
cargo build --release
```

### 2. Verify Environment and Embedded Model Readiness

```powershell
./target/release/sheaf doctor
```

If you also want the embedded `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` written into `./artifacts`, run:

```powershell
./target/release/sheaf init
```

### 3. Inspect Remaining `sorry` Theorems

```powershell
./target/release/sheaf check ../lean-workspace
```

`check` prints every `file:line: theorem` containing `sorry` and exits with code `1` when any remain (or `0` when all are proved).

### 4. Prove Theorems (Dry-Run vs. In-Place `--write`)

Run a dry-run first to inspect the 5-stage model telemetry (`Canonicalize`, `Tokenize`, `Lean-Sys1`, `Orion VDB`, `Lean Kernel`) and synthesized proof scripts without modifying files:

```powershell
./target/release/sheaf prove ../lean-workspace
```

Example live feedback output per theorem:

```text
[1/5] LeanWorkspace.add_zero_right (LeanWorkspace/Theorems.lean:5)
  ├─ 1. Canonicalize : _v0 : Nat ⊢ _v0 + 0 = _v0 [head_symbol=Eq]
  ├─ 2. Tokenize     : 9 tokens (pad_to=512)
  ├─ 3. Lean-Sys1    : strategy=ApplyLemma conf=0.58 noul=0.51 difficulty=Trivial gate=DirectExecute(ApplyLemma)
  ├─ 4. Orion VDB    : 7 hits -> top=["Nat.add_zero", ...]
  └─ 5. Lean Kernel  : VERIFIED in 1820ms [tactic: exact Nat.add_zero n]
```

Once satisfied, pass `--write` to update the `.lean` files in place:

```powershell
./target/release/sheaf prove ../lean-workspace --write
```

### Diagnostic Modes (`--gate-pct`, `--max-branches`, `--premise-tau`)

Three instant diagnostic modes are available from the command line:

- **Default balanced run:** `sheaf prove problem.lean --gate-pct 50 --max-branches 4`
- **Fast-path only (ablation):** `sheaf prove problem.lean --gate-pct 0` bypasses VDB fallback completely to measure pure System 1 accuracy.
- **Force VDB + branching (stress test):** `sheaf prove problem.lean --gate-pct 100 --max-branches 6` forces every theorem through HNSW + `PremiseFit` to benchmark retrieval recall and branch latency.

### Exit Codes & Error Diagnostics

- `0`: All discovered `sorry` theorems proved and kernel-verified (or `check` found zero `sorry`s).
- `1`: One or more theorems failed kernel verification (exact Lean 4 diagnostics are printed per theorem) or `check` found remaining `sorry`s.
- `2`: Invalid CLI usage or nonexistent `<PATH>`, or explicit failure in ONNX/VDB loading.

## CLI Reference

- `sheaf --help` prints all subcommands and flags.
- `sheaf --version` prints the binary version.
- `sheaf doctor` checks `lean --version`, embedded ONNX readiness, and `artifacts/` status.
- `sheaf init [--dir <PATH>]` writes the embedded `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` to disk.
- `sheaf check <PATH>` lists every remaining `sorry` theorem (`file:line: theorem`).
- `sheaf prove <PATH> [FLAGS]` runs the 5-stage model + VDB prover, Best-First tree search, and kernel verifier:
  - `--write` writes verified proofs to disk atomically at the end of the run.
  - `--gate-pct <0..100>` sets the `Lean-Sys1` confidence threshold (default `50.0`).
  - `--max-branches <1..16>` caps VDB-backed branches per step (default `4`).
  - `--premise-tau <0..1>` sets the minimum `PremiseFit` score (default `0.50`).
  - `--max-depth <1..16>` sets the maximum compositional search depth (default `4`).
  - `--max-nodes <1..1024>` caps expanded Best-First search nodes per theorem (default `64`).
  - `--max-workers <1..8>` sets concurrent Lean workers for multi-file worktree evaluation (default `2`).
  - `--max-heartbeats <>=10000>` sets per-probe `set_option maxHeartbeats` isolation (default `200000`).
  - `--strict-lint` enforces `unusedVariables` and `unusedSimpArgs` on the target theorem during `--write` verification.
  - `--plain` disables live TTY spinners for plain single-line `stderr` output.
  - `--json` emits NDJSON progress events on `stdout`.
  - `-v`, `--verbose` prints full 5-stage per-theorem telemetry.

## Verification

Run the full Rust workspace test suite:

```powershell
cargo test --workspace --release
```

## License & Author

Licensed under [AGPL-3.0-only](LICENSE).

Author: Steve Sujo Pulikottil
