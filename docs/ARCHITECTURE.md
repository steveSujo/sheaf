# Sheaf Architecture

Sheaf is built to run kernel-verified Lean 4 proof automation inside an **8 GB RAM / 4 GB VRAM** envelope as a single Rust binary, avoiding Python orchestration overhead and multi-GPU cloud dependencies.

## 1. Crate Topology

- `crates/lean-canonicalizer`: Alpha-renames local binders, normalizes dot-projections, collapses implicit instances, and extracts the goal head symbol (Eq, LT.lt, Exists, and others) so structurally identical goals share embeddings and cache keys.
- `crates/lean-sys1-ort`: Pure-Rust ONNX initializer loader and forward pass (LeanSys1OnnxSession) for `lean_sys1.onnx` (384-dim embeddings, 6 RoPE heads, RMSNorm, SwiGLU, masked attention pooling) plus 4d PremiseFit MLP scoring.
- `crates/lean-vdb`: LVDB v2 binary loader for `mathlib_sq8.vdb`, powered by [orion](https://github.com/steveSujo/orion) (pinned at `v0.1.0`) for scalar-quantized (SQ8) HNSW vector search, combined with symbol-to-module closure, import-DAG transitive reduction, and EpisodicCache.
- `crates/lean-repl`: Kernel verification helpers and ProofState JSON-RPC schema types.
- `crates/lean-prover`: CLI entry point, topological multi-file scheduler, content-addressed VFS (BlobStore + TreeSnapshot), tactic synthesis, Best-First tree search, batched `lean --json` / `lake env lean --json` evaluator, and 12 Hz indicatif terminal UI.

## 2. System 1 + System 2 Routing

Every discovered sorry theorem passes through five stages:

1. **Canonicalize:** Strips binder names and formats hypotheses and target into canonical tokens.
2. **Tokenize:** Splits delimiters identically to Python and frames with start/end tokens padded to 512.
3. **Lean-Sys1 Inference:** Runs LeanSys1OnnxSession forward inference and scales choice logits by T_choice (1.6071 in `temperatures.json`).
   - If the calibrated max probability meets the gate threshold (default 0.50), sheaf takes DirectExecute and probes strategy-ordered tactic candidates immediately.
   - If confidence drops below the threshold (for example on unseen compositions or open conjectures), sheaf takes FallbackToLeJepa.
4. **SQ8 HNSW Retrieval (`mathlib_sq8.vdb`):** Queries the state embedding against quantized premise vectors filtered by head symbol, scores surviving candidates through the PremiseFit 4d MLP, and branches on up to `--max-branches` premises.
5. **Best-First Search & Kernel Gate:** Expands search nodes up to `--max-depth` and `--max-nodes` with per-probe heartbeat isolation.

## 3. Content-Addressed VFS

Multi-file workspaces and multi-branch searches fork workspace state through immutable TreeSnapshot maps pointing into a shared BlobStore keyed by SHA-256 content hashes. Forking a branch is constant time without copying files on disk, dead-end branches drop cleanly without side effects, and `--write` flushes only dirty paths atomically after all theorems finish.

## 4. Two-Tier Loop Deduplication

Search avoids redundant kernel invocations at two levels:

- **Exact dedup (pre-inference):** Keyed by tree hash and canonical subgoal, short-circuiting repeated states before running Lean-Sys1.
- **Semantic dedup (post-inference):** EpisodicCache prunes subgoals whose normalized state embedding cosine similarity exceeds 0.98 against already-explored states in the same theorem search.

## 5. Strict Kernel & Axiom Gate

Every candidate proof is checked by the real Lean 4 kernel (`lean --json` or `lake env lean --json`) with `#print axioms` appended. Standard Lean axioms (propext, Classical.choice, Quot.sound) are allowed, while sorryAx or custom axioms are rejected unconditionally. Under `--write` with `--strict-lint`, sheaf runs up to two unusedSimpArgs pruning passes per occurrence before enforcing unusedVariables and unusedSimpArgs warnings scoped strictly to the target theorem's line span.
