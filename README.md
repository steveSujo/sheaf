# sheaf

A local-first, single-binary Lean 4 auto-prover written in Rust.

Sheaf embeds a quantized RoPE encoder, a scalar-quantized HNSW premise index, a content-addressed virtual filesystem, and a Lean 4 kernel verifier into a single ~25 MB executable. Everything runs offline in **8 GB RAM, CPU-only** (no GPU, no Python runtime, and no external model downloads).

[Demo](#30-second-self-verification-demo) • [Benchmarks](#empirical-benchmarks) • [Architecture](docs/ARCHITECTURE.md) • [How to Use](#how-to-use)

![sheaf live demo](assets/sheaf-demo.gif)


## Why sheaf?

Closing boilerplate simp, omega, and structural induction sorrys shouldn't require a cloud GPU or burning frontier LLM tokens in a compiler retry loop. Sheaf is an offline, CPU-only prover that closes routine Lean 4 goals locally on a standard laptop, whether you are proving specs by hand or offloading the grunt work from an LLM.

- **System 1 + System 2 routing:** A compact RoPE transformer predicts tactic strategy, difficulty, and calibrated confidence. High-confidence goals execute directly, while uncertain or out-of-distribution goals escalate dynamically to HNSW premise retrieval and Best-First tree search.
- **Content-addressed VFS:** Forks workspace states in constant time via SHA-256 hashes and shared strings so dead-end branches roll back cleanly and `--write` commits verified proofs atomically at exit.
- **Strict kernel and axiom gate:** Every candidate is verified by the real Lean 4 kernel with per-probe heartbeat caps and axiom checks, accepting proofs only under standard axioms (propext, Classical.choice, Quot.sound) with zero sorryAx.
- **Two-tier loop deduplication:** Short-circuits exact duplicate subgoals before inference and semantic loops in an episodic cache after inference.

For crate topology, VFS internals, and the SQ8 HNSW index layout, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What sheaf is (and isn't)

- **Good at:** Eliminating sorrys in algebraic laws, list and tree inductions, multi-module structure or state-machine invariants, and self-contained import Init specifications.
- **Not designed for:** Frontier research mathematics (IMO or deep Mathlib analysis) or open conjectures. When a goal is unprovable within `--max-nodes` or `--max-depth`, sheaf terminates cleanly with exit code 1 rather than hallucinating axioms.

## 30-Second Self-Verification Demo

The repository includes a standalone import Init specification at `examples/sheaf_self_verify/` modeling sheaf's own axiom gate, routing gate, and VFS invariants with 8 open sorry theorems:

```bash
# 1. Build the release binary (requires Rust + Lean 4.36.0 via elan)
cargo build --release

# 2. List the 8 open invariants (exits 1)
./target/release/sheaf check examples/sheaf_self_verify

# 3. Prove and kernel-verify all 8 invariants live (exits 0)
./target/release/sheaf prove examples/sheaf_self_verify

# 4. Write verified proofs in place with strict linter enforcement
./target/release/sheaf prove examples/sheaf_self_verify -w -s
```

## Empirical Benchmarks
Evaluated with `--release` on a single consumer laptop (Ryzen 5 4600H, 8 GB RAM, CPU-only, Lean 4.36.0) with zero `sorryAx` accepted:

| Suite | Scope | Proved |
| :--- | :--- | :---: |
| Single-module (`Test A1..A5`, `B1..B10`) | Arithmetic, `List`, `BinTree`, 3-record associativity | **15 / 15** (`159.7s`) |
| 4-folder diamond (`Test C1..C5`) | Cross-module import composition | **5 / 5** (`46.5s`) |
| 4-folder state-machine (`Test D1..D6`) | `structure` + `List` replay + dependent existentials | **6 / 6** (`51.8s`) |
| Open-hypothesis (`Test E1..E3`) | 2 provable + 1 unprovable conjecture (exits `1`) | **2 / 3** (`17.2s`) |
| Self-verification (`examples/sheaf_self_verify`) | `AxiomGate`, `GateSpec`, `VfsSpec` (`import Init`) | **8 / 8** (`38.2s`) |

## Operational Modes

- **Interactive TTY (default):** Live 12 Hz branch viewport on stderr, collapsing each proved theorem into a single permanent scrollback line.
- **Plain / CI (`--plain` or piped stderr):** Disables spinners and ANSI colors automatically (also respects NO_COLOR).
- **Verbose (`-v` / `--verbose`):** Prints the full 5-stage telemetry tree per theorem.
- **Machine JSON (`--json`):** Silences stderr and emits NDJSON events on stdout for editor integrations and benchmark harnesses.

## Reproducing Model Training & Calibration

End users building and running sheaf do **not** need Python, because the ONNX weights, tokenizer, calibration temperatures, and SQ8 premise index are embedded directly in the Rust binary. For researchers auditing the weights or out-of-distribution temperature calibration, see [scripts/README.md](scripts/README.md) for the standalone PyTorch training and ONNX export pipeline.

## How to Use

### Prerequisites & Setup

- **Lean 4.36.0** (lean and lake on PATH via elan)
- **Rust** (stable toolchain with cargo)

```bash
cargo build --release
./target/release/sheaf doctor          # Verify Lean toolchain & embedded assets
cargo test --workspace --release       # Run the full Rust test suite
```

<details>
<summary>CLI Reference: Subcommands, Key Flags, and Exit Codes</summary>

### Subcommands

- `sheaf doctor`: Checks `lean --version`, embedded ONNX readiness, and artifact status.
- `sheaf init [--dir <PATH>]`: Extracts embedded model and index assets to disk.
- `sheaf check [PATH]`: Lists every remaining sorry theorem (default: `.`).
- `sheaf prove [PATH] [FLAGS]`: Runs the prover and kernel verifier (default: `.`).

### Key Flags (`sheaf prove`)

- `-w`, `--write`: Commit verified proofs to disk atomically at exit.
- `-s`, `--strict-lint`: Enforce unusedVariables and unusedSimpArgs on the target theorem during `--write`.
- `-j`, `--max-workers <1..8>`: Concurrent Lean workers for multi-file evaluation (default 2).
- `--gate-pct <0..100>`: Confidence threshold for System 1 direct execution (default 50.0).
- `--max-branches <1..16>`: Max VDB-backed branches per step (default 4).
- `--premise-tau <0..1>`: Minimum premise-fit score (default 0.50).
- `--max-depth <1..16>` / `--max-nodes <1..1024>`: Best-First search bounds (defaults 4 / 64).
- `--max-heartbeats <>=10000>`: Per-probe Lean heartbeat cap (default 200000).
- `--plain` / `--json` / `-v` / `--verbose`: Output mode controls.

### Exit Codes

- 0: All discovered sorry theorems proved and kernel-verified (or `sheaf check` found zero sorrys).
- 1: One or more theorems failed kernel verification (or `sheaf check` found remaining sorrys).
- 2: Invalid CLI usage, missing path, or asset loading error.

</details>

## Contributing

Contributions are most useful in three areas:

- **Blind benchmark suites:** Multi-module Lean 4 specs, state machines, and algebraic data structures.
- **Tactic synthesis templates:** General-purpose existential, structural induction, and rewrite patterns.
- **Cross-platform testing:** Linux and macOS builds plus verification against newer Lean 4.x toolchains.

## License & Author

Licensed under [AGPL-3.0-only](LICENSE).

Author: Steve Sujo Pulikottil
