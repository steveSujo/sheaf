// SPDX-License-Identifier: AGPL-3.0-only
//! Model Context Protocol (MCP) newline-delimited JSON-RPC 2.0 stdio server for `sheaf`.

use crate::{
    DiscoveredTheorem, ProverError, compose_live_eval_source, discover_sorry_theorems,
    embedded_artifacts, extract_theorem_target, find_lake_workspace_root,
    index_workspace_lemmas_into_vdb, kernel_eval, normalize_fs_path,
    parse_unsolved_goal_from_diag, rerank_vdb_hits_lexical, search, synthesize_from_model_and_vdb,
    vfs,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const MCP_MAX_GOALS: usize = 3;
const MCP_MAX_ERROR_CHARS: usize = 300;
const MCP_MAX_DIAGNOSTICS: usize = 3;

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        return s.to_string();
    }
    if max_chars <= 3 {
        return s.chars().take(max_chars).collect();
    }
    let mut out: String = s.chars().take(max_chars - 3).collect();
    out.push_str("...");
    out
}

fn cap_goals_with_count(mut goals: Vec<String>) -> (Vec<String>, usize) {
    let total = goals.len();
    goals.truncate(MCP_MAX_GOALS);
    (goals, total)
}

fn cap_diagnostics(diags: Vec<String>) -> Vec<String> {
    diags
        .into_iter()
        .take(MCP_MAX_DIAGNOSTICS)
        .map(|d| {
            let first_line = d.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
            truncate_chars(first_line, MCP_MAX_ERROR_CHARS)
        })
        .collect()
}

struct McpRuntime {
    onnx_sess: lean_sys1_ort::LeanSys1OnnxSession,
    tokenizer_json: &'static str,
    temps: lean_sys1_ort::CalibrationTemperatures,
    base_vdb: lean_vdb::StaticVdbIndex,
}

/// Stateful MCP handler holding lazy-loaded ONNX/VDB assets and per-workspace VDB overlays.
#[derive(Default)]
pub struct McpServerState {
    runtime: Option<McpRuntime>,
    workspace_vdbs: HashMap<PathBuf, lean_vdb::StaticVdbIndex>,
}

impl McpServerState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_runtime(&mut self) -> Result<&McpRuntime, ProverError> {
        if self.runtime.is_none() {
            let emb = embedded_artifacts();
            let onnx_sess = lean_sys1_ort::LeanSys1OnnxSession::from_onnx_bytes(emb.lean_sys1_onnx)?;
            let temps: lean_sys1_ort::CalibrationTemperatures =
                serde_json::from_str(emb.temperatures_json)
                    .map_err(|e| ProverError::Repl(lean_repl::ReplError::Json(e)))?;
            let (base_vdb, _resolver) = lean_vdb::StaticVdbIndex::from_lvdb_bytes(emb.mathlib_sq8_vdb)?;
            self.runtime = Some(McpRuntime {
                onnx_sess,
                tokenizer_json: emb.tokenizer_json,
                temps,
                base_vdb,
            });
        }
        Ok(self.runtime.as_ref().expect("runtime initialized"))
    }

    fn vdb_for_workspace(
        &mut self,
        ws_key: &Path,
        theorems: &[DiscoveredTheorem],
    ) -> Result<lean_vdb::StaticVdbIndex, ProverError> {
        self.ensure_runtime()?;
        if let Some(cached) = self.workspace_vdbs.get(ws_key) {
            return Ok(cached.clone());
        }
        let rt = self.runtime.as_ref().expect("runtime initialized");
        let mut vdb = rt.base_vdb.clone();
        let mut blob_store = vfs::BlobStore::default();
        let mut files = BTreeMap::new();
        for thm in theorems {
            for dep in &thm.workspace_deps {
                if !files.contains_key(dep)
                    && let Ok(src) = std::fs::read_to_string(dep)
                {
                    files.insert(dep.clone(), blob_store.put(&src));
                }
            }
            if !files.contains_key(&thm.file)
                && let Ok(src) = std::fs::read_to_string(&thm.file)
            {
                files.insert(thm.file.clone(), blob_store.put(&src));
            }
        }
        if !files.is_empty() {
            index_workspace_lemmas_into_vdb(
                &files,
                &blob_store,
                &rt.onnx_sess,
                rt.tokenizer_json,
                &rt.temps,
                0.5,
                &mut vdb,
            );
        }
        self.workspace_vdbs.insert(ws_key.to_path_buf(), vdb.clone());
        Ok(vdb)
    }
}

fn jsonrpc_ok(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn jsonrpc_err(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message,
        }
    })
}

fn mcp_tool_result(payload: Value, is_error: bool) -> Value {
    let text = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    json!({
        "content": [{
            "type": "text",
            "text": text,
        }],
        "isError": is_error,
    })
}

fn mcp_tool_schemas() -> Value {
    json!({
        "tools": [
            {
                "name": "sheaf_goals",
                "description": "Inspect open sorry subgoals and VDB premises.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Path to Lean file" },
                        "theorem": { "type": "string", "description": "Theorem name" },
                        "top_k": { "type": "integer", "minimum": 1, "maximum": 32, "description": "Max premises (default 5)" },
                        "include_sig": { "type": "boolean", "description": "Include premise signatures (default false)" }
                    },
                    "required": ["file", "theorem"]
                }
            },
            {
                "name": "sheaf_try",
                "description": "Test batched tactics in memory via lean --server.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Path to Lean file" },
                        "theorem": { "type": "string", "description": "Theorem name" },
                        "tactics": { "type": "array", "items": { "type": "string" }, "description": "Tactic scripts to test in order" },
                        "tactic": { "type": "string", "description": "Single tactic fallback" },
                        "auto_close": { "type": "boolean", "description": "Auto-close open leaves (default true)" },
                        "timeout_ms": { "type": "integer", "minimum": 100, "maximum": 300000, "description": "Timeout in ms (default 15000)" }
                    },
                    "required": ["file", "theorem"]
                }
            },
            {
                "name": "sheaf_close",
                "description": "Run prefix tactic and local auto-closers.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Path to Lean file" },
                        "theorem": { "type": "string", "description": "Theorem name" },
                        "prefix_tactic": { "type": "string", "description": "Optional leading tactic" },
                        "timeout_ms": { "type": "integer", "minimum": 100, "maximum": 300000, "description": "Timeout in ms (default 15000)" }
                    },
                    "required": ["file", "theorem"]
                }
            },
            {
                "name": "sheaf_search",
                "description": "Search VDB premises for a goal.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "goal": { "type": "string", "description": "Lean 4 goal string" },
                        "top_k": { "type": "integer", "minimum": 1, "maximum": 32, "description": "Max premises (default 5)" },
                        "include_sig": { "type": "boolean", "description": "Include premise signatures (default false)" }
                    },
                    "required": ["goal"]
                }
            },
            {
                "name": "sheaf_apply",
                "description": "Kernel-verify proof and write to disk.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string", "description": "Path to Lean file" },
                        "theorem": { "type": "string", "description": "Theorem name" },
                        "proof_script": { "type": "string", "description": "Verified Lean 4 proof script" },
                        "verbose": { "type": "boolean", "description": "Include pruned proof text (default false)" }
                    },
                    "required": ["file", "theorem", "proof_script"]
                }
            }
        ]
    })
}

struct TheoremContext {
    norm_file: PathBuf,
    lake_root: Option<PathBuf>,
    thm: DiscoveredTheorem,
    full_name: String,
    disk_src: String,
    eval_src: String,
    all_theorems: Vec<DiscoveredTheorem>,
}

fn load_theorem_context(file: &str, theorem: &str) -> Result<TheoremContext, String> {
    let norm_file = normalize_fs_path(Path::new(file));
    if !norm_file.exists() {
        return Err(format!("file does not exist: {}", norm_file.display()));
    }
    let disk_src = std::fs::read_to_string(&norm_file)
        .map_err(|e| format!("failed to read {}: {e}", norm_file.display()))?;
    let all_theorems = discover_sorry_theorems(&norm_file)
        .map_err(|e| format!("failed to parse theorems in {}: {e}", norm_file.display()))?;
    let thm = all_theorems
        .iter()
        .find(|t| {
            let full = match &t.namespace {
                Some(ns) => format!("{ns}.{}", t.name),
                None => t.name.clone(),
            };
            t.name == theorem || full == theorem
        })
        .cloned()
        .ok_or_else(|| format!("sorry theorem '{theorem}' not found in {}", norm_file.display()))?;
    let full_name = match &thm.namespace {
        Some(ns) => format!("{ns}.{}", thm.name),
        None => thm.name.clone(),
    };
    let lake_root = find_lake_workspace_root(&norm_file);
    let base_root = lake_root.clone().unwrap_or_else(|| {
        norm_file
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    });
    let mut blob_store = vfs::BlobStore::default();
    let mut files = BTreeMap::new();
    for dep in &thm.workspace_deps {
        if let Ok(s) = std::fs::read_to_string(dep) {
            files.insert(dep.clone(), blob_store.put(&s));
        }
    }
    files.insert(norm_file.clone(), blob_store.put(&disk_src));
    let snap = vfs::TreeSnapshot::from_map(files);
    let eval_src = compose_live_eval_source(
        &snap,
        &blob_store,
        &norm_file,
        &thm.workspace_deps,
        &base_root,
    );
    Ok(TheoremContext {
        norm_file,
        lake_root,
        thm,
        full_name,
        disk_src,
        eval_src,
        all_theorems,
    })
}

fn query_vdb_premises(
    state: &mut McpServerState,
    ws_key: &Path,
    theorems: &[DiscoveredTheorem],
    canonical_goal: &str,
    head_symbol: &str,
    top_k: usize,
) -> Result<Vec<Value>, ProverError> {
    let vdb = state.vdb_for_workspace(ws_key, theorems)?;
    let rt = state.runtime.as_ref().expect("runtime initialized");
    let (ids, mask) = lean_sys1_ort::encode_canonical_text(rt.tokenizer_json, canonical_goal, 512)?;
    let (pred, _, _) = rt.onnx_sess.forward_with_gate(&ids, &mask, &rt.temps, 0.5)?;
    let query_emb = rt.onnx_sess.predict_lejepa_target(&pred.state_emb, pred.strategy);
    let mut hits = vdb.search_sq8_hnsw(&query_emb, head_symbol, top_k.max(16))?;
    hits.retain(|h| !kernel_eval::is_unknown_constant(&h.metadata.name));
    rerank_vdb_hits_lexical(&mut hits, canonical_goal);
    hits.truncate(top_k);
    Ok(hits
        .into_iter()
        .map(|h| {
            let score = 1.0 / (1.0 + h.distance.max(0.0));
            json!({
                "name": h.metadata.name,
                "signature": h.metadata.signature,
                "score": score,
            })
        })
        .collect())
}

fn handle_sheaf_goals(state: &mut McpServerState, args: &Value) -> Result<Value, Value> {
    let Some(file) = args.get("file").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'file'"));
    };
    let Some(theorem) = args.get("theorem").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'theorem'"));
    };
    let top_k = args
        .get("top_k")
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .clamp(1, 32) as usize;
    let include_sig = args
        .get("include_sig")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let ctx = match load_theorem_context(file, theorem) {
        Ok(c) => c,
        Err(msg) => return Ok(mcp_tool_result(json!({ "error": msg }), true)),
    };
    let deadline = Some(Instant::now() + Duration::from_secs(15));
    let goals = kernel_eval::query_plain_goals_for_script(
        &ctx.eval_src,
        &ctx.thm.name,
        &[],
        ctx.lake_root.as_deref(),
        deadline,
    )
    .unwrap_or_default();

    let target_str = extract_theorem_target(&ctx.thm.signature);
    let hyp_strs: Vec<String> = ctx
        .thm
        .binders
        .iter()
        .map(|(n, t)| format!("{n} : {t}"))
        .collect();
    let collapsed = lean_canonicalizer::collapse_implicit_instances(target_str);
    let (norm_hyps, norm_target) = lean_canonicalizer::normalize_alpha_vars(&hyp_strs, &collapsed);
    let head = lean_canonicalizer::extract_head_symbol(&norm_target)
        .map(|h| h.as_str().to_string())
        .unwrap_or_else(|_| "Eq".to_string());
    let canonical = format!("{} ⊢ {}", norm_hyps.join(", "), norm_target);
    let ws_key = ctx
        .lake_root
        .clone()
        .unwrap_or_else(|| ctx.norm_file.parent().unwrap_or(Path::new(".")).to_path_buf());

    let vdb_premises: Vec<Value> = match query_vdb_premises(
        state,
        &ws_key,
        &ctx.all_theorems,
        &canonical,
        &head,
        top_k,
    ) {
        Ok(v) => v
            .into_iter()
            .map(|p| {
                if include_sig {
                    json!({ "name": p["name"], "signature": p["signature"], "score": p["score"] })
                } else {
                    json!({ "name": p["name"], "score": p["score"] })
                }
            })
            .collect(),
        Err(e) => return Ok(mcp_tool_result(json!({ "error": truncate_chars(&e.to_string(), MCP_MAX_ERROR_CHARS) }), true)),
    };

    let (goals, remaining_count) = cap_goals_with_count(goals);
    Ok(mcp_tool_result(
        json!({
            "goals": goals,
            "remaining_count": remaining_count,
            "local_defs": ctx.thm.local_defs,
            "vdb_premises": vdb_premises,
        }),
        false,
    ))
}

fn run_local_close(
    state: &mut McpServerState,
    ctx: &TheoremContext,
    prefix_lines: &[String],
    deadline: Option<Instant>,
) -> Result<Option<Vec<String>>, String> {
    let ws_key = ctx
        .lake_root
        .clone()
        .unwrap_or_else(|| ctx.norm_file.parent().unwrap_or(Path::new(".")).to_path_buf());
    let vdb = state
        .vdb_for_workspace(&ws_key, &ctx.all_theorems)
        .map_err(|e| e.to_string())?;
    let rt = state.runtime.as_ref().expect("runtime initialized");

    let target_str = extract_theorem_target(&ctx.thm.signature);
    let hyp_strs: Vec<String> = ctx
        .thm
        .binders
        .iter()
        .map(|(n, t)| format!("{n} : {t}"))
        .collect();
    let collapsed = lean_canonicalizer::collapse_implicit_instances(target_str);
    let (norm_hyps, norm_target) = lean_canonicalizer::normalize_alpha_vars(&hyp_strs, &collapsed);
    let head_symbol = lean_canonicalizer::extract_head_symbol(&norm_target)
        .map(|h| h.as_str().to_string())
        .unwrap_or_else(|_| "Eq".to_string());
    let canonical = format!("{} ⊢ {}", norm_hyps.join(", "), norm_target);

    let (ids, mask) = lean_sys1_ort::encode_canonical_text(rt.tokenizer_json, &canonical, 512)
        .map_err(|e| e.to_string())?;
    let (pred, _, _) = rt
        .onnx_sess
        .forward_with_gate(&ids, &mask, &rt.temps, 0.5)
        .map_err(|e| e.to_string())?;
    let query_emb = rt.onnx_sess.predict_lejepa_target(&pred.state_emb, pred.strategy);
    let mut vdb_hits = vdb
        .search_sq8_hnsw(&query_emb, &head_symbol, 16)
        .unwrap_or_default();
    vdb_hits.retain(|h| !kernel_eval::is_unknown_constant(&h.metadata.name));
    rerank_vdb_hits_lexical(&mut vdb_hits, &canonical);
    let top_lemmas: Vec<String> = vdb_hits
        .iter()
        .take(6)
        .map(|h| h.metadata.name.clone())
        .collect();

    let base_cands = synthesize_from_model_and_vdb(
        &ctx.thm,
        &head_symbol,
        &pred,
        &vdb_hits,
        &top_lemmas,
        8,
        false,
    );
    let candidates: Vec<Vec<String>> = if prefix_lines.is_empty() {
        base_cands
    } else {
        base_cands
            .into_iter()
            .map(|c| {
                let mut combined = prefix_lines.to_vec();
                combined.extend(c);
                combined
            })
            .collect()
    };

    let ui_session = crate::ui::UiSession::start(crate::ui::UiMode::Plain);
    let ui = ui_session.handle();
    let (solved, _diag, unsolved) = kernel_eval::verify_candidates_batched_with_deadline_and_ui(
        &ctx.eval_src,
        &ctx.thm,
        &ctx.full_name,
        &candidates,
        ctx.lake_root.as_deref(),
        200_000,
        &ui,
        0,
        true,
        false,
        deadline,
    )
    .map_err(|e| e.to_string())?;

    if let Some((_win_src, win_script)) = solved {
        return Ok(Some(win_script));
    }

    if deadline.is_some_and(|dl| Instant::now() < dl) && !unsolved.is_empty() {
        let mut blob_store = vfs::BlobStore::default();
        let mut map = BTreeMap::new();
        map.insert(ctx.norm_file.clone(), blob_store.put(&ctx.eval_src));
        let root_tree = vfs::TreeSnapshot::from_map(map);
        let search_res = search::run_best_first_tree_search(
            &mut blob_store,
            root_tree,
            ctx.norm_file.clone(),
            canonical.clone(),
            3,
            16,
            |store, node, dedup| {
                if deadline.is_some_and(|dl| Instant::now() >= dl) {
                    return Ok(search::StepOutcome::Pruned("timeout exceeded".into()));
                }
                let cur_src = node
                    .tree
                    .read_file(store, &node.target_file)
                    .as_deref()
                    .map(str::to_string)
                    .unwrap_or_else(|| ctx.eval_src.clone());
                let step_cands: Vec<Vec<String>> = if node.depth == 0 {
                    let mut seeds: Vec<Vec<String>> =
                        unsolved.iter().take(3).map(|(s, _)| s.clone()).collect();
                    for prog in ["ext", "funext", "intro", "constructor"] {
                        let mut s = prefix_lines.to_vec();
                        s.push(prog.to_string());
                        seeds.push(s);
                    }
                    seeds
                } else {
                    let Ok((s_ids, s_mask)) = lean_sys1_ort::encode_canonical_text(
                        rt.tokenizer_json,
                        &node.open_subgoal,
                        512,
                    ) else {
                        return Ok(search::StepOutcome::Pruned("tokenize subgoal failed".into()));
                    };
                    let Ok((s_pred, _, _)) =
                        rt.onnx_sess
                            .forward_with_gate(&s_ids, &s_mask, &rt.temps, 0.5)
                    else {
                        return Ok(search::StepOutcome::Pruned("sys1 subgoal failed".into()));
                    };
                    if !dedup.record_sys1_and_check_tier2_with_progress(
                        &s_pred.state_emb,
                        node.parent_subgoal.as_deref(),
                        &node.open_subgoal,
                    ) {
                        return Ok(search::StepOutcome::Pruned("loop pruned".into()));
                    }
                    let mut sub_closers = Vec::new();
                    if !ctx.thm.local_defs.is_empty() {
                        let defs = ctx.thm.local_defs.join(", ");
                        sub_closers.push(format!("dsimp only [{defs}]"));
                        sub_closers.push(format!("simp_all [{defs}]"));
                    }
                    if ctx.thm.has_mathlib {
                        sub_closers.push(
                            "all_goals (first | field_simp at * <;> (first | linarith | nlinarith | positivity) | rify at *; field_simp at * <;> (first | linarith | nlinarith | positivity) | ring | linarith | nlinarith | norm_num)".to_string(),
                        );
                    }
                    sub_closers.push(
                        "all_goals (first | rfl | simp_all | omega | grind | tauto)".to_string(),
                    );
                    sub_closers
                        .into_iter()
                        .map(|c| {
                            let mut s = node.partial_script.clone();
                            if let Some(last) = s.last_mut() {
                                *last = format!("{last}; {c}");
                            } else {
                                s.push(c);
                            }
                            s
                        })
                        .collect()
                };
                let (step_solved, step_diag, step_unsolved) =
                    kernel_eval::verify_candidates_batched_with_deadline_and_ui(
                        &cur_src,
                        &ctx.thm,
                        &ctx.full_name,
                        &step_cands,
                        ctx.lake_root.as_deref(),
                        200_000,
                        &ui,
                        node.id * 4,
                        true,
                        false,
                        deadline,
                    )
                    .map_err(|e| e.to_string())?;
                if let Some((win_src, win_script)) = step_solved {
                    let next_tree = node
                        .tree
                        .with_updates(store, &[(node.target_file.clone(), win_src)]);
                    return Ok(search::StepOutcome::Solved {
                        tree: next_tree,
                        script: win_script,
                    });
                }
                let mut children = Vec::new();
                for (cand_script, gdiag) in step_unsolved.into_iter().take(4) {
                    if deadline.is_some_and(|dl| Instant::now() >= dl) {
                        break;
                    }
                    if let Some((sh, st)) = parse_unsolved_goal_from_diag(&gdiag) {
                        let sub_canon = format!("{} ⊢ {}", sh.join(", "), st);
                        let next_tree = node.tree.with_updates(
                            store,
                            &[(node.target_file.clone(), cur_src.clone())],
                        );
                        children.push(search::SearchNode {
                            id: 0,
                            parent: Some(node.id),
                            depth: node.depth + 1,
                            tree: next_tree,
                            target_file: node.target_file.clone(),
                            open_subgoal: sub_canon,
                            parent_subgoal: Some(node.open_subgoal.clone()),
                            partial_script: cand_script,
                            priority: search::compute_node_priority(
                                pred.difficulty,
                                pred.answer_confidence,
                                node.depth + 1,
                            ),
                        });
                    }
                }
                if children.is_empty() {
                    Ok(search::StepOutcome::Pruned(step_diag))
                } else {
                    Ok(search::StepOutcome::Progressed(children))
                }
            },
        );
        if let Some(win_script) = search_res.winning_script {
            return Ok(Some(win_script));
        }
    }
    Ok(None)
}

fn handle_sheaf_try(state: &mut McpServerState, args: &Value) -> Result<Value, Value> {
    let Some(file) = args.get("file").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'file'"));
    };
    let Some(theorem) = args.get("theorem").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'theorem'"));
    };
    let mut tactics: Vec<String> = args
        .get("tactics")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if tactics.is_empty()
        && let Some(single) = args.get("tactic").and_then(Value::as_str)
    {
        let trimmed = single.trim();
        if !trimmed.is_empty() {
            tactics.push(trimmed.to_string());
        }
    }
    if tactics.is_empty() {
        return Err(jsonrpc_err(
            Value::Null,
            -32602,
            "missing required parameter 'tactics'",
        ));
    }
    let auto_close = args
        .get("auto_close")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let timeout_ms = args
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .unwrap_or(15_000)
        .clamp(100, 300_000);

    let ctx = match load_theorem_context(file, theorem) {
        Ok(c) => c,
        Err(msg) => {
            return Ok(mcp_tool_result(
                json!({ "error": truncate_chars(&msg, MCP_MAX_ERROR_CHARS) }),
                true,
            ));
        }
    };
    let t0 = Instant::now();
    let deadline = Some(t0 + Duration::from_millis(timeout_ms));

    let mut best_progress: Option<Vec<String>> = None;
    let mut last_error: Option<(Vec<String>, Vec<String>)> = None;

    for tactic in &tactics {
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            return Ok(mcp_tool_result(
                json!({
                    "proved": false,
                    "remaining_goals": [],
                    "remaining_count": 0,
                    "diagnostics": ["timeout exceeded"],
                    "elapsed_ms": t0.elapsed().as_millis() as u64,
                }),
                true,
            ));
        }
        let outcome = match kernel_eval::eval_tactic_in_memory(
            &ctx.eval_src,
            &ctx.thm.name,
            tactic,
            ctx.lake_root.as_deref(),
            deadline,
        ) {
            Ok(o) => o,
            Err(e) => {
                return Ok(mcp_tool_result(
                    json!({ "error": truncate_chars(&e.to_string(), MCP_MAX_ERROR_CHARS) }),
                    true,
                ));
            }
        };
        if outcome.timed_out {
            return Ok(mcp_tool_result(
                json!({
                    "proved": false,
                    "remaining_goals": [],
                    "remaining_count": 0,
                    "diagnostics": ["timeout exceeded"],
                    "elapsed_ms": t0.elapsed().as_millis() as u64,
                }),
                true,
            ));
        }
        if outcome.proved {
            return Ok(mcp_tool_result(
                json!({
                    "proved": true,
                    "proof_script": tactic,
                    "remaining_goals": [],
                    "remaining_count": 0,
                    "diagnostics": [],
                    "elapsed_ms": t0.elapsed().as_millis() as u64,
                }),
                false,
            ));
        }
        if outcome.diagnostics.is_empty() {
            if auto_close {
                let tier1 = format!(
                    "{tactic}\nall_goals (first | rfl | omega | simp_all | ring | linarith)"
                );
                if let Ok(t1_out) = kernel_eval::eval_tactic_in_memory(
                    &ctx.eval_src,
                    &ctx.thm.name,
                    &tier1,
                    ctx.lake_root.as_deref(),
                    deadline,
                ) && t1_out.proved
                {
                    return Ok(mcp_tool_result(
                        json!({
                            "proved": true,
                            "proof_script": tier1,
                            "remaining_goals": [],
                            "remaining_count": 0,
                            "diagnostics": [],
                            "elapsed_ms": t0.elapsed().as_millis() as u64,
                        }),
                        false,
                    ));
                }
                let prefix_lines: Vec<String> = tactic
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect();
                if let Ok(Some(win_script)) = run_local_close(state, &ctx, &prefix_lines, deadline)
                {
                    return Ok(mcp_tool_result(
                        json!({
                            "proved": true,
                            "proof_script": win_script.join("\n"),
                            "remaining_goals": [],
                            "remaining_count": 0,
                            "diagnostics": [],
                            "elapsed_ms": t0.elapsed().as_millis() as u64,
                        }),
                        false,
                    ));
                }
            }
            let should_replace = best_progress
                .as_ref()
                .is_none_or(|prev_goals| outcome.remaining_goals.len() < prev_goals.len());
            if should_replace {
                best_progress = Some(outcome.remaining_goals);
            }
        } else {
            last_error = Some((outcome.remaining_goals, outcome.diagnostics));
        }
    }

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    if let Some(remaining_goals) = best_progress {
        let (capped, remaining_count) = cap_goals_with_count(remaining_goals);
        return Ok(mcp_tool_result(
            json!({
                "proved": false,
                "remaining_goals": capped,
                "remaining_count": remaining_count,
                "diagnostics": [],
                "elapsed_ms": elapsed_ms,
            }),
            false,
        ));
    }
    let (remaining_goals, diagnostics) = last_error.unwrap_or_default();
    let (capped, remaining_count) = cap_goals_with_count(remaining_goals);
    Ok(mcp_tool_result(
        json!({
            "proved": false,
            "remaining_goals": capped,
            "remaining_count": remaining_count,
            "diagnostics": cap_diagnostics(diagnostics),
            "elapsed_ms": elapsed_ms,
        }),
        true,
    ))
}

fn handle_sheaf_close(state: &mut McpServerState, args: &Value) -> Result<Value, Value> {
    let Some(file) = args.get("file").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'file'"));
    };
    let Some(theorem) = args.get("theorem").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'theorem'"));
    };
    let prefix_tactic = args.get("prefix_tactic").and_then(Value::as_str);
    let timeout_ms = args
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .unwrap_or(15_000)
        .clamp(100, 300_000);

    let ctx = match load_theorem_context(file, theorem) {
        Ok(c) => c,
        Err(msg) => {
            return Ok(mcp_tool_result(
                json!({ "error": truncate_chars(&msg, MCP_MAX_ERROR_CHARS) }),
                true,
            ));
        }
    };
    let t0 = Instant::now();
    let deadline = Some(t0 + Duration::from_millis(timeout_ms));

    let prefix_lines: Vec<String> = prefix_tactic
        .map(|p| {
            p.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        })
        .unwrap_or_default();

    if !prefix_lines.is_empty() {
        let try_res = kernel_eval::eval_tactic_in_memory(
            &ctx.eval_src,
            &ctx.thm.name,
            &prefix_lines.join("\n"),
            ctx.lake_root.as_deref(),
            deadline,
        );
        match try_res {
            Ok(out) if out.timed_out => {
                return Ok(mcp_tool_result(
                    json!({
                        "proved": false,
                        "remaining_goals": [],
                        "diagnostics": ["timeout exceeded"],
                        "elapsed_ms": t0.elapsed().as_millis() as u64,
                    }),
                    true,
                ));
            }
            Ok(out) if out.proved => {
                return Ok(mcp_tool_result(
                    json!({
                        "proved": true,
                        "proof_script": prefix_lines.join("\n"),
                        "remaining_goals": [],
                        "elapsed_ms": t0.elapsed().as_millis() as u64,
                    }),
                    false,
                ));
            }
            Ok(out) if !out.diagnostics.is_empty() => {
                let (capped, remaining_count) = cap_goals_with_count(out.remaining_goals);
                return Ok(mcp_tool_result(
                    json!({
                        "proved": false,
                        "remaining_goals": capped,
                        "remaining_count": remaining_count,
                        "diagnostics": cap_diagnostics(out.diagnostics),
                        "elapsed_ms": t0.elapsed().as_millis() as u64,
                    }),
                    true,
                ));
            }
            Err(e) => {
                return Ok(mcp_tool_result(
                    json!({ "error": truncate_chars(&e.to_string(), MCP_MAX_ERROR_CHARS) }),
                    true,
                ));
            }
            Ok(_) => {}
        }
    }

    match run_local_close(state, &ctx, &prefix_lines, deadline) {
        Ok(Some(win_script)) => Ok(mcp_tool_result(
            json!({
                "proved": true,
                "proof_script": win_script.join("\n"),
                "remaining_goals": [],
                "remaining_count": 0,
                "elapsed_ms": t0.elapsed().as_millis() as u64,
            }),
            false,
        )),
        Ok(None) => {
            let remaining_goals = kernel_eval::query_plain_goals_for_script(
                &ctx.eval_src,
                &ctx.thm.name,
                &prefix_lines,
                ctx.lake_root.as_deref(),
                Some(Instant::now() + Duration::from_secs(3)),
            )
            .unwrap_or_default();
            let (capped, remaining_count) = cap_goals_with_count(remaining_goals);
            Ok(mcp_tool_result(
                json!({
                    "proved": false,
                    "remaining_goals": capped,
                    "remaining_count": remaining_count,
                    "elapsed_ms": t0.elapsed().as_millis() as u64,
                }),
                false,
            ))
        }
        Err(msg) => Ok(mcp_tool_result(
            json!({ "error": truncate_chars(&msg, MCP_MAX_ERROR_CHARS) }),
            true,
        )),
    }
}

fn handle_sheaf_search(state: &mut McpServerState, args: &Value) -> Result<Value, Value> {
    let Some(goal) = args.get("goal").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'goal'"));
    };
    let top_k = args
        .get("top_k")
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .clamp(1, 32) as usize;
    let include_sig = args
        .get("include_sig")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let cgoal = match lean_canonicalizer::canonicalize_goal(&[], goal) {
        Ok(g) => g,
        Err(e) => return Ok(mcp_tool_result(json!({ "error": e.to_string() }), true)),
    };
    let canonical = format!("{} ⊢ {}", cgoal.hypotheses.join(", "), cgoal.target);
    let head = cgoal.head_symbol.as_str().to_string();

    if let Err(e) = state.ensure_runtime() {
        return Ok(mcp_tool_result(json!({ "error": e.to_string() }), true));
    }
    let vdb = state
        .workspace_vdbs
        .values()
        .next()
        .cloned()
        .unwrap_or_else(|| state.runtime.as_ref().unwrap().base_vdb.clone());
    let rt = state.runtime.as_ref().unwrap();
    let (ids, mask) = match lean_sys1_ort::encode_canonical_text(rt.tokenizer_json, &canonical, 512)
    {
        Ok(v) => v,
        Err(e) => return Ok(mcp_tool_result(json!({ "error": e.to_string() }), true)),
    };
    let (pred, _, _) = match rt.onnx_sess.forward_with_gate(&ids, &mask, &rt.temps, 0.5) {
        Ok(v) => v,
        Err(e) => return Ok(mcp_tool_result(json!({ "error": e.to_string() }), true)),
    };
    let query_emb = rt.onnx_sess.predict_lejepa_target(&pred.state_emb, pred.strategy);
    let mut hits = match vdb.search_sq8_hnsw(&query_emb, &head, top_k.max(16)) {
        Ok(h) => h,
        Err(e) => return Ok(mcp_tool_result(json!({ "error": e.to_string() }), true)),
    };
    hits.retain(|h| !kernel_eval::is_unknown_constant(&h.metadata.name));
    rerank_vdb_hits_lexical(&mut hits, &canonical);
    hits.truncate(top_k);
    let premises: Vec<Value> = hits
        .into_iter()
        .map(|h| {
            let score = 1.0 / (1.0 + h.distance.max(0.0));
            if include_sig {
                json!({
                    "name": h.metadata.name,
                    "signature": h.metadata.signature,
                    "score": score,
                })
            } else {
                json!({
                    "name": h.metadata.name,
                    "score": score,
                })
            }
        })
        .collect();
    Ok(mcp_tool_result(json!({ "premises": premises }), false))
}

fn handle_sheaf_apply(args: &Value) -> Result<Value, Value> {
    let Some(file) = args.get("file").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'file'"));
    };
    let Some(theorem) = args.get("theorem").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'theorem'"));
    };
    let Some(proof_script) = args.get("proof_script").and_then(Value::as_str) else {
        return Err(jsonrpc_err(Value::Null, -32602, "missing required parameter 'proof_script'"));
    };
    let verbose = args
        .get("verbose")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let ctx = match load_theorem_context(file, theorem) {
        Ok(c) => c,
        Err(msg) => {
            return Ok(mcp_tool_result(
                json!({ "ok": false, "diagnostics": cap_diagnostics(vec![msg]) }),
                true,
            ));
        }
    };
    let script_lines: Vec<String> = proof_script
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    if script_lines.is_empty() {
        return Ok(mcp_tool_result(
            json!({ "ok": false, "diagnostics": ["empty proof_script"] }),
            true,
        ));
    }
    let deadline = Some(Instant::now() + Duration::from_secs(30));
    match kernel_eval::verify_and_prune_for_apply(
        &ctx.disk_src,
        &ctx.thm.name,
        &ctx.full_name,
        &script_lines,
        ctx.lake_root.as_deref(),
        deadline,
    ) {
        Ok(Ok((updated_disk_src, pruned_script, axioms))) => {
            if let Err(e) = std::fs::write(&ctx.norm_file, updated_disk_src) {
                return Ok(mcp_tool_result(
                    json!({ "ok": false, "diagnostics": cap_diagnostics(vec![format!("write failed: {e}")]) }),
                    true,
                ));
            }
            let payload = if verbose {
                json!({
                    "ok": true,
                    "axioms": axioms,
                    "pruned_proof": pruned_script.join("\n"),
                })
            } else {
                json!({
                    "ok": true,
                    "axioms": axioms,
                })
            };
            Ok(mcp_tool_result(payload, false))
        }
        Ok(Err(diags)) => Ok(mcp_tool_result(
            json!({
                "ok": false,
                "axioms": [],
                "diagnostics": cap_diagnostics(diags),
            }),
            true,
        )),
        Err(e) => Ok(mcp_tool_result(
            json!({
                "ok": false,
                "axioms": [],
                "diagnostics": cap_diagnostics(vec![e.to_string()]),
            }),
            true,
        )),
    }
}

/// Process a single JSON-RPC 2.0 line for `sheaf mcp`. Returns `None` for notifications.
pub fn handle_mcp_jsonrpc_line(state: &mut McpServerState, line: &str) -> Option<Value> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let req: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => {
            return Some(jsonrpc_err(Value::Null, -32700, &format!("parse error: {e}")));
        }
    };
    let id = req.get("id").cloned();
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    let Some(req_id) = id else {
        // Notification (e.g. `notifications/initialized`): no reply per JSON-RPC 2.0.
        return None;
    };

    match method {
        "initialize" => Some(jsonrpc_ok(
            req_id,
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "sheaf",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }),
        )),
        "ping" => Some(jsonrpc_ok(req_id, json!({}))),
        "tools/list" => Some(jsonrpc_ok(req_id, mcp_tool_schemas())),
        "tools/call" => {
            let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
            let tool_name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            let res = match tool_name {
                "sheaf_goals" => handle_sheaf_goals(state, &args),
                "sheaf_try" => handle_sheaf_try(state, &args),
                "sheaf_close" => handle_sheaf_close(state, &args),
                "sheaf_search" => handle_sheaf_search(state, &args),
                "sheaf_apply" => handle_sheaf_apply(&args),
                _ => Err(jsonrpc_err(
                    req_id.clone(),
                    -32602,
                    &format!("unknown tool '{tool_name}'"),
                )),
            };
            match res {
                Ok(payload) => Some(jsonrpc_ok(req_id, payload)),
                Err(mut err_val) => {
                    if let Some(obj) = err_val.as_object_mut() {
                        obj.insert("id".to_string(), req_id);
                    }
                    Some(err_val)
                }
            }
        }
        _ => Some(jsonrpc_err(
            req_id,
            -32601,
            &format!("method not found: '{method}'"),
        )),
    }
}

/// Run the blocking newline-delimited JSON-RPC 2.0 MCP server on `stdin`/`stdout`.
pub fn run_mcp_stdio_server() -> Result<(), ProverError> {
    let mut state = McpServerState::new();
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line_res in stdin.lock().lines() {
        let line = line_res.map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))?;
        if let Some(reply) = handle_mcp_jsonrpc_line(&mut state, &line) {
            let encoded = serde_json::to_string(&reply)
                .map_err(|e| ProverError::Repl(lean_repl::ReplError::Json(e)))?;
            writeln!(out, "{encoded}")
                .and_then(|_| out.flush())
                .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_handshake_and_tools_list() {
        let mut state = McpServerState::new();
        let init_reply = handle_mcp_jsonrpc_line(
            &mut state,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        )
        .expect("initialize must reply");
        assert_eq!(init_reply["id"], 1);
        assert_eq!(init_reply["result"]["serverInfo"]["name"], "sheaf");
        assert!(state.runtime.is_none(), "runtime must lazy-load after initialize");

        assert!(
            handle_mcp_jsonrpc_line(
                &mut state,
                r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            )
            .is_none()
        );

        let ping_reply = handle_mcp_jsonrpc_line(
            &mut state,
            r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#,
        )
        .expect("ping must reply");
        assert_eq!(ping_reply["result"], json!({}));

        let list_reply = handle_mcp_jsonrpc_line(
            &mut state,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#,
        )
        .expect("tools/list must reply");
        let tools = list_reply["result"]["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 5);
        for name in [
            "sheaf_goals",
            "sheaf_try",
            "sheaf_close",
            "sheaf_search",
            "sheaf_apply",
        ] {
            assert!(
                tools.iter().any(|t| t["name"] == name && t["inputSchema"]["type"] == "object"),
                "missing schema for {name}"
            );
        }
    }

    #[test]
    fn test_mcp_try_and_apply_flow() {
        let dir = std::env::temp_dir().join("sheaf_test_mcp_flow");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("McpFlow.lean");
        let orig = "theorem mcp_add_zero (n : Nat) : n + 0 = n := by\n  sorry\n";
        std::fs::write(&file, orig).unwrap();

        let mut state = McpServerState::new();
        let goals_req = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "sheaf_goals",
                "arguments": { "file": file.to_string_lossy(), "theorem": "mcp_add_zero" }
            }
        })
        .to_string();
        let goals_reply = handle_mcp_jsonrpc_line(&mut state, &goals_req).unwrap();
        assert_eq!(goals_reply["result"]["isError"], false);
        let goals_body: Value =
            serde_json::from_str(goals_reply["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert!(!goals_body["goals"].as_array().unwrap().is_empty());

        let bad_req = json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "sheaf_try",
                "arguments": {
                    "file": file.to_string_lossy(),
                    "theorem": "mcp_add_zero",
                    "tactics": ["exact bogus_nonexistent"],
                    "auto_close": false
                }
            }
        })
        .to_string();
        let bad_reply = handle_mcp_jsonrpc_line(&mut state, &bad_req).unwrap();
        assert_eq!(bad_reply["result"]["isError"], true);
        let bad_body: Value =
            serde_json::from_str(bad_reply["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(bad_body["proved"], false);
        assert!(!bad_body["diagnostics"].as_array().unwrap().is_empty());
        assert_eq!(std::fs::read_to_string(&file).unwrap(), orig);

        let ok_req = json!({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "sheaf_try",
                "arguments": {
                    "file": file.to_string_lossy(),
                    "theorem": "mcp_add_zero",
                    "tactics": ["exact bogus_nonexistent", "simp"]
                }
            }
        })
        .to_string();
        let ok_reply = handle_mcp_jsonrpc_line(&mut state, &ok_req).unwrap();
        assert_eq!(ok_reply["result"]["isError"], false);
        let ok_text = ok_reply["result"]["content"][0]["text"].as_str().unwrap();
        assert!(!ok_text.contains('\n'), "expected minified JSON");
        let ok_body: Value = serde_json::from_str(ok_text).unwrap();
        assert_eq!(ok_body["proved"], true);
        assert!(ok_body["remaining_goals"].as_array().unwrap().is_empty());

        let apply_req = json!({
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "sheaf_apply",
                "arguments": {
                    "file": file.to_string_lossy(),
                    "theorem": "mcp_add_zero",
                    "proof_script": "simp"
                }
            }
        })
        .to_string();
        let apply_reply = handle_mcp_jsonrpc_line(&mut state, &apply_req).unwrap();
        assert_eq!(apply_reply["result"]["isError"], false);
        let apply_body: Value =
            serde_json::from_str(apply_reply["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(apply_body["ok"], true);
        assert!(apply_body.get("pruned_proof").is_none());
        let disk_after = std::fs::read_to_string(&file).unwrap();
        assert!(!disk_after.contains("sorry"));
        assert!(disk_after.contains("simp"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mcp_batched_try_and_auto_close() {
        let (capped, count) =
            cap_goals_with_count(vec!["g1".into(), "g2".into(), "g3".into(), "g4".into()]);
        assert_eq!(capped.len(), 3);
        assert_eq!(count, 4);
        let long_err = format!("{}\nverbose trace line", "x".repeat(450));
        let capped_diags = cap_diagnostics(vec![long_err]);
        assert_eq!(capped_diags[0].chars().count(), 300);
        assert!(capped_diags[0].ends_with("..."));

        let dir = std::env::temp_dir().join("sheaf_test_mcp_batched_ac");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("BatchedAc.lean");
        std::fs::write(
            &file,
            "theorem mcp_ac (n : Nat) : n = n -> n + 0 = n := by\n  sorry\n",
        )
        .unwrap();

        let mut state = McpServerState::new();

        // 1) Partial progress with auto_close=false -> isError: false, proved: false, non-empty goals
        let partial_req = json!({
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "sheaf_try",
                "arguments": {
                    "file": file.to_string_lossy(),
                    "theorem": "mcp_ac",
                    "tactics": ["intro _h"],
                    "auto_close": false
                }
            }
        })
        .to_string();
        let partial_reply = handle_mcp_jsonrpc_line(&mut state, &partial_req).unwrap();
        assert_eq!(partial_reply["result"]["isError"], false);
        let partial_text = partial_reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap();
        assert!(!partial_text.contains('\n'), "expected minified JSON");
        let partial_body: Value = serde_json::from_str(partial_text).unwrap();
        assert_eq!(partial_body["proved"], false);
        assert_eq!(partial_body["remaining_count"], 1);
        assert_eq!(partial_body["remaining_goals"].as_array().unwrap().len(), 1);
        assert!(partial_body.get("tactic").is_none());

        // 2) Batched with default auto_close=true -> stops on first prove
        let ac_req = json!({
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "sheaf_try",
                "arguments": {
                    "file": file.to_string_lossy(),
                    "theorem": "mcp_ac",
                    "tactics": ["exact bogus", "intro _h"]
                }
            }
        })
        .to_string();
        let ac_reply = handle_mcp_jsonrpc_line(&mut state, &ac_req).unwrap();
        assert_eq!(ac_reply["result"]["isError"], false);
        let ac_body: Value =
            serde_json::from_str(ac_reply["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(ac_body["proved"], true);
        assert!(ac_body["remaining_goals"].as_array().unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
