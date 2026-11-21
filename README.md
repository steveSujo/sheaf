# leanProof

`leanProof` is a local-first compositional Lean 4 auto-prover built in Rust and Python. It keeps memory and VRAM usage inside an 8 GB RAM / 4 GB VRAM envelope by pairing an embedded `Lean-Sys1` ONNX router, an embedded `orion` HNSW + SQ8 vector index, and a real Lean 4 kernel verifier.

The compiled `lean-prover` binary carries its own embedded `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` assets so it runs out of the box without external downloads.

## Prerequisites

- **Lean 4.36.0** (`lean` and `lake` on `PATH` via `elan`)
- **Rust** (stable toolchain with `cargo`)
- **uv** (for Python preflight, Colab training prep, and verification scripts)

## Quickstart

Check that Lean 4 is reachable and inspect the embedded model status:

```powershell
cargo run -- doctor
cargo run -- init
```

Run the prover against a `.lean` file or a `lake` workspace directory (add `--write` to replace `sorry` in place after kernel verification):

```powershell
cargo run -- prove ../lean-workspace
cargo run -- prove ../lean-workspace --write
cargo run -- check ../lean-workspace
```

## CLI Reference

- `cargo run -- --help` prints all subcommands and flags.
- `cargo run -- doctor` checks `lean --version`, embedded ONNX readiness, and `artifacts/` status.
- `cargo run -- init [--dir <PATH>]` writes the embedded `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` to disk.
- `cargo run -- prove <PATH> [--write]` discovers `sorry` theorems, synthesizes proofs, verifies `#print axioms` (zero `sorryAx`), and optionally writes them back.
- `cargo run -- check <PATH>` scans for remaining `sorry` theorems and exits non-zero if any remain.

## Python Pre-Flight & Tests

Run the lightweight pre-flight check and Compute Unit estimator on any clean `uv` environment:

```powershell
uv run python -m lean_composer.preflight
```

Run the full Rust and Python test suites (install CPU-only `torch` and `onnx` via `--group cpu-test`):

```powershell
cargo test --workspace
uv run --group cpu-test pytest
```
