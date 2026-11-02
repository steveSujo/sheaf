# leanProof

`leanProof` is a local-first compositional Lean 4 auto-prover built in Rust. It keeps memory and VRAM usage inside an 8 GB RAM / 4 GB VRAM envelope by pairing an embedded `Lean-Sys1` ONNX router, an embedded `orion` HNSW + SQ8 vector index, and a real Lean 4 kernel verifier.

The compiled `lean-prover` binary carries its own embedded `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` assets so it runs out of the box without external downloads or Python dependencies.

## Prerequisites

- **Lean 4.36.0** (`lean` and `lake` on `PATH` via `elan`)
- **Rust** (stable toolchain with `cargo`)

## Quickstart

Build the release binary, verify Lean 4 and embedded model readiness, and prove a `.lean` file or `lake` workspace:

```powershell
cargo build --release
./target/release/lean-prover doctor
./target/release/lean-prover check ../lean-workspace
./target/release/lean-prover prove ../lean-workspace --write
```

You can also invoke the CLI directly via `cargo run`:

```powershell
cargo run -- doctor
cargo run -- prove ../lean-workspace
```

## CLI Reference

- `lean-prover --help` prints all subcommands and flags.
- `lean-prover doctor` checks `lean --version`, embedded ONNX readiness, and `artifacts/` status.
- `lean-prover init [--dir <PATH>]` writes the embedded `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` to disk.
- `lean-prover prove <PATH> [--write]` discovers `sorry` theorems, synthesizes proofs, verifies `#print axioms` (zero `sorryAx`), prints each proof script, and optionally writes them back.
- `lean-prover check <PATH>` lists every remaining `sorry` theorem (`file:line: theorem`) and exits non-zero if any remain.

## Verification

Run the full Rust workspace test suite:

```powershell
cargo test --workspace
```
