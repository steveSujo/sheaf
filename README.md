# leanProof

`leanProof` is a local-first compositional Lean 4 auto-prover built in Rust and Python. It keeps memory and VRAM usage inside an 8 GB RAM / 4 GB VRAM envelope by pairing a small calibrated `Lean-Sys1` ONNX router, an embedded `orion` HNSW + SQ8 vector index, and a real Lean 4 kernel verifier.

## Prerequisites

- **Lean 4.36.0** (`lean` and `lake` on `PATH` via `elan`)
- **Rust** (stable toolchain with `cargo`)
- **uv** (for Python preflight, training prep, and verification scripts)

## Quickstart

Check that Lean 4 is reachable and bootstrap the baseline runtime artifacts:

```powershell
cargo run -- doctor
cargo run -- init
```

Run the prover against a `.lean` file or a `lake` workspace directory (add `--write` to replace `sorry` in place after kernel verification):

```powershell
cargo run -- prove ../lean-workspace --write
cargo run -- check ../lean-workspace
```

## CLI Reference

- `cargo run -- --help` prints all subcommands and flags.
- `cargo run -- doctor` checks `lean --version` and `artifacts/` readiness.
- `cargo run -- init [--dir <PATH>]` writes baseline `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb`.
- `cargo run -- prove <PATH> [--write]` discovers `sorry` theorems, synthesizes proofs, verifies `#print axioms` (zero `sorryAx`), and optionally writes them back.
- `cargo run -- check <PATH>` scans for remaining `sorry` theorems and exits non-zero if any remain.

## Python Pre-Flight & Colab Training

Run the local pre-flight check and Compute Unit estimator before any Colab GPU run:

```powershell
uv run python -m lean_composer.preflight
```

Run the full Rust and Python test suites:

```powershell
cargo test --workspace
uv run pytest
```
