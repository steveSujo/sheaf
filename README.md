# sheaf

**A local-first, single-binary Lean 4 auto-prover in Rust.**

`sheaf` embeds its own quantized RoPE encoder (`lean_sys1.onnx`), scalar-quantized HNSW premise index (`mathlib_sq8.vdb`), content-addressed VFS, and Lean 4 kernel verifier into a single ~25 MB executable. Everything runs inside an **8 GB RAM / 4 GB VRAM** envelope with zero Python runtime or external model downloads.

```text
$ ./target/release/sheaf prove examples/sheaf_self_verify --plain
✓ SheafSpec.AxiomGate.verify_empty       DirectExecute(AtomicClose)  1.8s [simp [verifyAllowedAxioms]]
✓ SheafSpec.AxiomGate.verify_rejects     DirectExecute(AtomicClose)  2.1s [simp [verifyAllowedAxioms]]
...
Proved 8/8 theorems in 16.4s (8 DirectExecute, 0 Escalated, 0 sorryAx)
```

---

## Why `sheaf`?

Most neural theorem provers assume multi-GPU cloud clusters, Python orchestration layers, and gigabytes of VRAM. `sheaf` is built for the opposite constraint: fast, offline, kernel-verified proof automation on a consumer laptop.

- **System 1 + System 2 Routing:** A compact RoPE transformer (`Lean-Sys1`) predicts tactic strategy, difficulty, and calibrated confidence (`T_choice`). High-confidence goals execute directly, while uncertain or out-of-distribution goals escalate dynamically to SQ8 HNSW premise retrieval (`PremiseFit`) and Best-First tree search.
- **Content-Addressed VFS (`BlobStore` + `TreeSnapshot`):** Forks workspace states in $O(1)$ via SHA-256 hashes and `Arc<str>` sharing so dead-end branches roll back cleanly and `--write` commits verified proofs atomically at exit.
- **Strict Kernel & Axiom Gate:** Every candidate is verified by the real Lean 4 kernel with per-probe `set_option maxHeartbeats` isolation and `#print axioms` checks, accepting proofs only under `propext`, `Classical.choice`, and `Quot.sound` with zero `sorryAx`.
- **Two-Tier Loop Deduplication:** Short-circuits exact `(tree_hash, canon_subgoal)` duplicates before inference and semantic loops ($\cos > 0.98$) in `EpisodicCache` after inference.

---

## What `sheaf` Is (and Isn't)

- **Good at:** Eliminating `sorry`s in algebraic laws, `List`/`BinTree` inductions, multi-module `structure` or state-machine invariants, and self-contained `import Init` specifications.
- **Not designed for:** Frontier research mathematics (IMO or deep Mathlib analysis) or open conjectures. When a goal is unprovable within `--max-nodes` or `--max-depth`, `sheaf` terminates cleanly with exit code `1` rather than hallucinating axioms.

---

## 30-Second Self-Verification Demo

The repo includes a standalone `import Init` specification at `examples/sheaf_self_verify/` modeling `sheaf`'s own `AxiomGate`, `GateSpec`, and `VfsSpec` invariants with 8 open `sorry` theorems:

```bash
# 1. Build the release binary (requires Rust + Lean 4.36.0 via elan)
cargo build --release

# 2. List the 8 open invariants (exits 1)
./target/release/sheaf check examples/sheaf_self_verify

# 3. Prove and kernel-verify all 8 invariants live (exits 0)
./target/release/sheaf prove examples/sheaf_self_verify

# 4. Write verified proofs in place with linter enforcement
./target/release/sheaf prove examples/sheaf_self_verify --write --strict-lint
```

---

## Empirical Benchmarks (`--release`)

Evaluated on a single consumer laptop (Ryzen 5 4600H, 8 GB RAM, Lean 4.36.0) across blind single-module, multi-folder, and open-hypothesis suites:

| Suite | Scope | Proved | Notes |
| --- | --- | --- | --- |
| Single-module (`Test A1..A5`, `B1..B10`) | Arithmetic, `List`, `BinTree`, 3-record associativity | **15 / 15** | Zero `sorryAx` |
| 4-folder diamond (`Test C1..C5`) | Cross-module import composition | **5 / 5** | Per-occurrence `unusedSimpArgs` pruning on `--write` |
| 4-folder state-machine (`Test D1..D6`) | `structure` + `List` replay + dependent `∃` | **6 / 6** | Dynamic `DirectExecute` to VDB escalation |
| Open-hypothesis anti-cheat (`Test E1..E3`) | 2 provable + 1 open conjecture | **2 / 3** | Bounded termination (`exit 1`), zero `sorryAx` |
| Self-verification (`examples/sheaf_self_verify`) | `AxiomGate`, `GateSpec`, `VfsSpec` | **8 / 8** | Standalone `import Init` spec |

---

## Operational Modes

- **Interactive TTY (default):** 12 Hz `indicatif` live branch viewport on `stderr`, collapsing each proved theorem into a single permanent scrollback line.
- **Plain / CI (`--plain` or piped `stderr`):** Disables spinners and ANSI escapes automatically.
- **Verbose (`-v` / `--verbose`):** Prints the full 5-stage telemetry tree (`Canonicalize` -> `Tokenize` -> `Lean-Sys1` -> `VDB` -> `Lean Kernel`).
- **Machine JSON (`--json`):** Silences `stderr` and emits clean NDJSON events on `stdout` for editor integrations and benchmark harnesses.

---

## Reproducing Model Training & Calibration

End users building and running `sheaf` do **not** need Python, because `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` are embedded directly in the Rust binary. For researchers auditing the weights or `T_choice = 1.6071` OOD calibration, the standalone PyTorch training and ONNX export pipeline lives in `scripts/`:

- [scripts/requirements-train.txt](scripts/requirements-train.txt): Pinned training and ONNX export dependencies.
- [scripts/colab_stage1_train.py](scripts/colab_stage1_train.py): RoPE + RMSNorm + SwiGLU encoder, `SIGReg` regularization, and `PremiseFit` 4d MLP pre-training.
- [scripts/colab_stage3_4_tune_and_promote.py](scripts/colab_stage3_4_tune_and_promote.py): `label_smoothing=0.075` fine-tuning, mixed ID+OOD L-BFGS temperature calibration, and ONNX `heads.premise_fit.*` export verification.

---

## How to Use

### Prerequisites & Setup

- **Lean 4.36.0** (`lean` and `lake` on `PATH` via `elan`)
- **Rust** (stable toolchain with `cargo`)

```bash
cargo build --release
./target/release/sheaf doctor       # Verify Lean toolchain & embedded ONNX/VDB assets
cargo test --workspace --release    # Run the full Rust test suite
```

<details>
<summary>CLI Reference: Subcommands, Key Flags, and Exit Codes</summary>

### Subcommands

- `sheaf doctor`: Checks `lean --version`, embedded ONNX readiness, and `artifacts/` status.
- `sheaf init [--dir <PATH>]`: Extracts embedded `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` to disk.
- `sheaf check <PATH>`: Lists every remaining `sorry` theorem (`file:line: theorem`).
- `sheaf prove <PATH> [FLAGS]`: Runs the prover and kernel verifier.

### Key Flags (`sheaf prove`)

- `--write`: Commit verified proofs to disk atomically at exit.
- `--strict-lint`: Enforce `unusedVariables` and `unusedSimpArgs` on the target theorem during `--write` verification.
- `--gate-pct <0..100>`: Confidence threshold for System 1 direct execution (default `50.0`, `0` = pure System 1 ablation, `100` = force VDB + branching).
- `--max-branches <1..16>`: Max VDB-backed branches per step (default `4`).
- `--premise-tau <0..1>`: Minimum `PremiseFit` score (default `0.50`).
- `--max-depth <1..16>` / `--max-nodes <1..1024>`: Best-First search bounds (defaults `4` / `64`).
- `--max-workers <1..8>`: Concurrent Lean workers for multi-file worktree evaluation (default `2`).
- `--max-heartbeats <>=10000>`: Per-probe `set_option maxHeartbeats` cap (default `200000`).
- `--plain` / `--json` / `-v, --verbose`: Output mode controls.

### Exit Codes

- `0`: All discovered `sorry` theorems proved and kernel-verified (or `check` found zero `sorry`s).
- `1`: One or more theorems failed kernel verification (or `check` found remaining `sorry`s).
- `2`: Invalid CLI usage, missing `<PATH>`, or asset loading error.

</details>

---

## License & Author

Licensed under [AGPL-3.0-only](LICENSE).

Author: Steve Sujo Pulikottil
