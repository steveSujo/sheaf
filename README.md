# sheaf

A token-saving Lean 4 MCP server and local CPU-only prover written in Rust.

Sheaf cuts LLM compiler-loop token burn via a persistent `lean --server` (~35ms warm probes), local leaf auto-close, and delta-only `$/lean/plainGoal` output. It embeds a quantized RoPE encoder, a scalar-quantized HNSW premise index, a content-addressed virtual filesystem, and a Lean 4 kernel verifier into a single ~25 MB executable that runs offline in **8 GB RAM, CPU-only** with no GPU or Python runtime.

[MCP Setup](#mcp-server-for-llm-agents) • [Demo](#30-second-self-verification-demo) • [Benchmarks](#empirical-benchmarks) • [Architecture](docs/ARCHITECTURE.md) • [How to Use](#how-to-use)

![sheaf live demo](assets/sheaf-demo.gif)


## Why sheaf?

Closing boilerplate `simp`, `omega`, and structural induction sorrys shouldn't require a cloud GPU or burning frontier LLM tokens in a compiler retry loop. Sheaf pairs fast local proving with an MCP server so frontier models supply proof structure while trivial subgoals and syntax checks resolve locally.

- **Token-saving agent loop:** The LLM writes high-level strategy (`have` or `induction`), while `sheaf` tests batched tactics in ~35ms, auto-closes trivial leaves locally, and returns compact goal deltas only without echoing full files or verbose logs.
- **Persistent `lean --server` + live `$/lean/plainGoal`:** Pays import load once per workspace so warm `didChange` probes avoid cold CLI restarts.
- **System 1 + System 2 routing with hybrid VDB:** A compact RoPE transformer predicts tactic strategy, difficulty, and calibrated confidence. Uncertain or out-of-distribution goals escalate to **hybrid lexical + SQ8 HNSW** retrieval plus a **dynamic workspace VDB overlay** that indexes the project's own proved lemmas automatically.
- **Content-addressed VFS:** Forks workspace states in constant time via SHA-256 hashes and shared strings so dead-end branches roll back cleanly and `--write` commits verified proofs atomically at exit.
- **Strict kernel and axiom gate:** Every candidate is verified by the real Lean 4 kernel with per-probe heartbeat caps and axiom checks, accepting proofs only under standard axioms (`propext`, `Classical.choice`, `Quot.sound`) with zero `sorryAx`.
- **Two-tier loop deduplication:** Short-circuits exact duplicate subgoals before inference and semantic loops in an episodic cache after inference.

For crate topology, VFS internals, and the SQ8 HNSW index layout, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## MCP Server for LLM Agents

Add `sheaf mcp` to Claude Code, Cursor, Zed, or Neovim:

```json
{
  "mcpServers": {
    "sheaf": { "command": "sheaf", "args": ["mcp"] }
  }
}
```

| Tool | Purpose | Key Input / Output |
| :--- | :--- | :--- |
| `sheaf_goals` | Sync file into warm `lean --server` and inspect open goal | `{ file, theorem, top_k? }` -> `{ goals, local_defs, vdb_premises }` |
| `sheaf_try` | Test batched `tactics[]` + `auto_close` in memory (~35ms) | `{ file, theorem, tactics[], auto_close?, timeout_ms? }` -> `{ proved, remaining_goals, diagnostics }` |
| `sheaf_close` | Apply optional `prefix_tactic` and auto-close leaves locally | `{ file, theorem, prefix_tactic?, timeout_ms? }` -> `{ proved, proof_script?, remaining_goals }` |
| `sheaf_search` | Canonicalize goal and query merged embedded + workspace VDB | `{ goal, top_k? }` -> `{ premises }` |
| `sheaf_apply` | Kernel-verify (`#print axioms` + `unusedSimpArgs` prune) before disk write | `{ file, theorem, proof_script }` -> `{ applied, pruned_proof?, axioms }` |

Token-saving mechanics:
- Tiny tool schemas so tool definitions add minimal prompt overhead.
- Capped `remaining_goals` (max 3) and truncated errors (~300 chars) to keep compiler feedback tight.
- Minified JSON responses and read-only in-memory evaluation until `sheaf_apply` passes kernel verification.

## What sheaf is (and isn't)

- **Standalone:** Closes offline boilerplate, algebraic laws, list and tree inductions, multi-module state-machine invariants, and self-contained `import Init` specs.
- **MCP-guided:** Pairs with your LLM for hard Mathlib or multi-step proofs without burning tokens on syntax retries or leaf cleanup.
- **Not designed for:** Unassisted frontier research mathematics or open conjectures. When a goal is unprovable within `--max-nodes` or `--max-depth`, sheaf terminates cleanly with exit code 1 rather than hallucinating axioms.

## 30-Second Self-Verification Demo

The repository includes a standalone `import Init` specification at `examples/sheaf_self_verify/` modeling sheaf's own axiom gate, routing gate, and VFS invariants with 8 open `sorry` theorems:

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
Standalone CPU-only baseline evaluated with `--release` on a single consumer laptop (Ryzen 5 4600H, 8 GB RAM, CPU-only, Lean 4.36.0) with zero `sorryAx` accepted:

| Suite | Scope | Proved |
| :--- | :--- | :---: |
| Single-module (`Test A1..A5`, `B1..B10`) | Arithmetic, `List`, `BinTree`, 3-record associativity | **15 / 15** (`159.7s`) |
| 4-folder diamond (`Test C1..C5`) | Cross-module import composition | **5 / 5** (`46.5s`) |
| 4-folder state-machine (`Test D1..D6`) | `structure` + `List` replay + dependent existentials | **6 / 6** (`51.8s`) |
| Open-hypothesis (`Test E1..E3`) | 2 provable + 1 unprovable conjecture (exits `1`) | **2 / 3** (`17.2s`) |
| Self-verification (`examples/sheaf_self_verify`) | `AxiomGate`, `GateSpec`, `VfsSpec` (`import Init`) | **8 / 8** (`38.2s`) |

MCP token-savings comparison (reserved for upcoming end-to-end measurement):

| Workflow | Total Tokens | Compiler Turns | Notes |
| :--- | :---: | :---: | :--- |
| Raw LLM + `lake build` retries | *TBD* | *TBD* | Cold CLI restarts and full diagnostics |
| LLM + `sheaf mcp` | *TBD* | *TBD* | Persistent `lean --server`, delta goals, local leaf close |

## Operational Modes

- **Interactive TTY (default):** Live 12 Hz branch viewport on stderr, collapsing each proved theorem into a single permanent scrollback line.
- **Plain / CI (`--plain` or piped stderr):** Disables spinners and ANSI colors automatically (also respects `NO_COLOR`).
- **Verbose (`-v` / `--verbose`):** Prints the full 5-stage telemetry tree per theorem and surfaces the lowest-goal-count stuck goal state on failures or OOD budget exhaustion.
- **Machine JSON (`--json`):** Silences stderr and emits NDJSON events on stdout (`TheoremFinished` includes `status`, `vdb_hits`, and `stuck_goal`, and `--diff --json` emits `WorkspaceDiff` events without breaking NDJSON parsers).
- **MCP Stdio (`sheaf mcp`):** Newline-delimited JSON-RPC 2.0 over stdin/stdout with strict `stdout` hygiene, lazy ONNX/VDB init for instant handshake, and strict per-call timeout kill/reset.

## Reproducing Model Training & Calibration

End users building and running sheaf do **not** need Python, because the ONNX weights, tokenizer, calibration temperatures, and SQ8 premise index are embedded directly in the Rust binary. For researchers auditing the weights or out-of-distribution temperature calibration, see [scripts/README.md](scripts/README.md) for the standalone PyTorch training and ONNX export pipeline.

## How to Use

### Prerequisites & Setup

- **Lean 4.36.0** (`lean` and `lake` on `PATH` via `elan`)
- **Rust** (stable toolchain with `cargo`)

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
- `sheaf check [PATH] [--diagnose]`: Lists every remaining sorry theorem (default: `.`). Passing `--diagnose` runs a zero-ONNX pre-flight build and sanity probe to classify `Trivial`, `ValidOpen`, `FalseSpec`, or compile errors in under 2 seconds.
- `sheaf prove [PATH] [FLAGS]`: Runs the prover and kernel verifier (default: `.`).
- `sheaf mcp`: Starts the newline-delimited JSON-RPC 2.0 MCP stdio server for LLM agents.

### Key Flags (`sheaf prove`)

- `-k`, `--filter <PATTERN>`: Prove only theorems whose name matches `<PATTERN>` (warns and exits `1` if zero theorems match).
- `-w`, `--write`: Commit verified proofs to disk atomically at exit.
- `--diff`: Print a unified diff of synthesized proofs without modifying files (emits a `WorkspaceDiff` NDJSON event when combined with `--json`).
- `-s`, `--strict-lint`: Enforce `unusedVariables` and `unusedSimpArgs` pruning and verification (including compound `simp_all [...]; grind` scripts).
- `--ood-timeout <1..300>`: Per-theorem escalation budget in seconds for out-of-distribution goals (default `15`).
- `-j`, `--max-workers <1..8>`: Concurrent Lean workers for multi-file evaluation (default 2).
- `--gate-pct <0..100>`: Confidence threshold for System 1 direct execution (default 50.0).
- `--max-branches <1..16>`: Max VDB-backed branches per step (default 4).
- `--premise-tau <0..1>`: Minimum premise-fit score (default 0.50).
- `--max-depth <1..16>` / `--max-nodes <1..1024>`: Best-First search bounds (defaults 4 / 64).
- `--max-heartbeats <>=10000>`: Per-probe Lean heartbeat cap (default 200000).
- `--plain` / `--json` / `-v` / `--verbose`: Output mode controls.

### Exit Codes

- 0: All discovered sorry theorems proved and kernel-verified (or `sheaf check` found zero sorrys).
- 1: One or more theorems failed kernel verification, `-k` matched zero theorems, or `sheaf check` found remaining sorrys.
- 2: Invalid CLI usage, missing path, asset loading error, or Lean compile error.
- 3: `FalseSpec` detected during pre-flight sanity probing (`sheaf check --diagnose` or `sheaf prove`).

</details>

## Contributing

Contributions are most useful in four areas:

- **VDB corpus expansion:** Indexing additional open-source Lean 4 and Mathlib4 theorem corpora into the SQ8 HNSW index.
- **General search & LeJEPA improvements:** Better Best-First search heuristics, calibration, and next-state embedding prediction.
- **MCP client recipes:** Tested configurations and workflows for Claude Code, Cursor, Zed, Neovim, and other MCP clients.
- **Blind benchmark suites & cross-platform testing:** Multi-module Lean 4 specs, Linux/macOS builds, and verification against newer Lean 4.x toolchains.

## Acknowledgments

- **Lean 4 & Lean FRO:** For the Lean 4 kernel, `lake` build system, and `lean --server` (`didChange`, `$/lean/plainGoal`) verification toolchain.
- **Mathlib4:** For the open-source theorem and premise corpus indexed in `mathlib_sq8.vdb`.
- **LeJEPA:** Architectural inspiration for `FallbackToLeJepa` next-state embedding prediction and SIGReg regularization.
- **Upstream Rust ecosystem:** For the open-source crates compiled into `sheaf` (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full crate names, versions, and SPDX licenses).

## License & Author

Licensed under [AGPL-3.0-only](LICENSE).

Author: Steve Sujo Pulikottil
