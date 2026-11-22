//! Hybrid Kernel Evaluator: Single-file batched probes with `maxHeartbeats` isolation
//! and Multi-file ephemeral dirty-worktree evaluation with early cancellation.

use crate::ui::{BranchStatus, UiEvent, UiHandle};
use crate::vfs::{BlobStore, TreeSnapshot};
use crate::{DiscoveredTheorem, ProverError, replace_theorem_sorry};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Result of batched candidate evaluation: `(solved_src_and_script, last_diag, unsolved_goal_diags)`.
pub type BatchedVerifyResult = (
    Option<(String, Vec<String>)>,
    String,
    Vec<(Vec<String>, String)>,
);

/// Abstraction over Lean 4 kernel verification backends (`BatchLeanCliBackend` or future persistent `PantographSession`).
pub trait KernelBackend {
    fn eval_batch(
        &self,
        src: &str,
        thm: &DiscoveredTheorem,
        full_name: &str,
        candidates: &[Vec<String>],
        lake_root: Option<&Path>,
        max_heartbeats: usize,
    ) -> Result<BatchedVerifyResult, ProverError>;

    fn audit_winner(
        &self,
        src: &str,
        thm_name: &str,
        full_name: &str,
        script: &[String],
        lake_root: Option<&Path>,
    ) -> Result<Result<String, String>, ProverError>;
}

/// Default `KernelBackend` using batched `lake env lean --json` probes with per-probe `maxHeartbeats`.
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchLeanCliBackend;

impl KernelBackend for BatchLeanCliBackend {
    fn eval_batch(
        &self,
        src: &str,
        thm: &DiscoveredTheorem,
        full_name: &str,
        candidates: &[Vec<String>],
        lake_root: Option<&Path>,
        max_heartbeats: usize,
    ) -> Result<BatchedVerifyResult, ProverError> {
        verify_candidates_batched_with_heartbeats(
            src,
            thm,
            full_name,
            candidates,
            lake_root,
            max_heartbeats,
        )
    }

    fn audit_winner(
        &self,
        src: &str,
        thm_name: &str,
        full_name: &str,
        script: &[String],
        lake_root: Option<&Path>,
    ) -> Result<Result<String, String>, ProverError> {
        audit_single_candidate(src, thm_name, full_name, script, lake_root)
    }
}

pub(crate) fn run_lean_json(
    file: &Path,
    lake_root: Option<&Path>,
) -> Result<std::process::Output, ProverError> {
    if let Some(root) = lake_root {
        std::process::Command::new("lake")
            .args(["env", "lean", "--json"])
            .arg(file)
            .current_dir(root)
            .output()
    } else {
        std::process::Command::new("lean")
            .arg("--json")
            .arg(file)
            .output()
    }
    .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))
}

pub(crate) fn spawn_lean_json_piped(
    file: &Path,
    lake_root: Option<&Path>,
) -> Result<std::process::Child, ProverError> {
    if let Some(root) = lake_root {
        std::process::Command::new("lake")
            .args(["env", "lean", "--json"])
            .arg(file)
            .current_dir(root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
    } else {
        std::process::Command::new("lean")
            .arg("--json")
            .arg(file)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
    }
    .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))
}

/// Audit a single candidate script by replacing `sorry` and running `#print axioms` through `verify_allowed_axioms`.
pub fn audit_single_candidate(
    src: &str,
    thm_name: &str,
    full_name: &str,
    script: &[String],
    lake_root: Option<&Path>,
) -> Result<Result<String, String>, ProverError> {
    let body = script
        .iter()
        .map(|l| format!("  {l}"))
        .collect::<Vec<_>>()
        .join("\n");
    let candidate_src = replace_theorem_sorry(src, thm_name, &body);
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp_file = std::env::temp_dir().join(format!(
        "sheaf_audit_{thm_name}_{unique}_{:?}.lean",
        std::thread::current().id()
    ));
    let audit = format!("{candidate_src}\n\n#print axioms {full_name}\n");
    std::fs::write(&tmp_file, &audit)
        .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))?;
    let out = run_lean_json(&tmp_file, lake_root)?;
    let _ = std::fs::remove_file(&tmp_file);

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut has_error = false;
    let mut illegal_axiom: Option<String> = None;
    let mut first_error_msg: Option<String> = None;

    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let sev = v.get("severity").and_then(|s| s.as_str()).unwrap_or("");
            let data = v.get("data").and_then(|d| d.as_str()).unwrap_or("");
            if sev == "error" {
                has_error = true;
                if first_error_msg.is_none() && !data.is_empty() {
                    first_error_msg = Some(data.to_string());
                }
            }
            if (data.contains("depends on axioms:")
                || data.contains("does not depend on any axioms"))
                && let Err(e) = lean_repl::verify_allowed_axioms(data)
            {
                illegal_axiom = Some(e.to_string());
            }
        }
    }

    if out.status.success() && !has_error && illegal_axiom.is_none() {
        Ok(Ok(candidate_src))
    } else {
        let diag = illegal_axiom
            .or(first_error_msg)
            .or_else(|| {
                stdout
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "kernel rejected".to_string());
        Ok(Err(diag))
    }
}

fn strip_trailing_docstrings_and_attrs(prefix: &str) -> &str {
    let mut s = prefix.trim_end();
    loop {
        if s.ends_with("-/")
            && let Some(start) = s.rfind("/--")
        {
            s = s[..start].trim_end();
            continue;
        }
        if let Some(last_nl) = s.rfind('\n') {
            let last_line = s[last_nl + 1..].trim();
            if last_line.starts_with("@[") && last_line.ends_with(']') {
                s = s[..last_nl].trim_end();
                continue;
            }
        } else {
            let trimmed = s.trim();
            if trimmed.starts_with("@[") && trimmed.ends_with(']') {
                s = "";
                continue;
            }
        }
        break;
    }
    s
}

/// Evaluate candidate scripts in batches of 8 with per-probe `set_option maxHeartbeats <N> in` isolation.
pub fn verify_candidates_batched_with_heartbeats(
    src: &str,
    thm: &DiscoveredTheorem,
    full_name: &str,
    candidates: &[Vec<String>],
    lake_root: Option<&Path>,
    max_heartbeats: usize,
) -> Result<BatchedVerifyResult, ProverError> {
    verify_candidates_batched_with_heartbeats_and_ui(
        src,
        thm,
        full_name,
        candidates,
        lake_root,
        max_heartbeats,
        &UiHandle::noop(),
        0,
    )
}

/// Evaluate candidate scripts in batches of 8, streaming `lake env lean --json` diagnostics live to `ui`.
#[allow(clippy::too_many_arguments)]
pub fn verify_candidates_batched_with_heartbeats_and_ui(
    src: &str,
    thm: &DiscoveredTheorem,
    full_name: &str,
    candidates: &[Vec<String>],
    lake_root: Option<&Path>,
    max_heartbeats: usize,
    ui: &UiHandle,
    branch_offset: usize,
) -> Result<BatchedVerifyResult, ProverError> {
    let mut last_diag = String::from("model produced no valid kernel-verified tactic");
    let mut unsolved_diags: Vec<(Vec<String>, String)> = Vec::new();
    if candidates.is_empty() {
        return Ok((None, last_diag, unsolved_diags));
    }

    let needle = format!("theorem {}", thm.name);
    let Some(thm_pos) = src.find(&needle) else {
        for (idx, script) in candidates.iter().enumerate() {
            let branch_id = branch_offset + idx + 1;
            let preview = script.join("; ");
            ui.send(UiEvent::BranchSpawned {
                id: branch_id,
                tag: "Probe".to_string(),
                preview: preview.clone(),
            });
            ui.send(UiEvent::BranchUpdated {
                id: branch_id,
                status: BranchStatus::Running,
                elapsed_ms: 0,
            });
            let t0 = Instant::now();
            match audit_single_candidate(src, &thm.name, full_name, script, lake_root)? {
                Ok(cand_src) => {
                    ui.send(UiEvent::BranchUpdated {
                        id: branch_id,
                        status: BranchStatus::Verified { tactic: preview },
                        elapsed_ms: t0.elapsed().as_millis() as u64,
                    });
                    return Ok((
                        Some((cand_src, script.clone())),
                        String::new(),
                        unsolved_diags,
                    ));
                }
                Err(d) => {
                    ui.send(UiEvent::BranchUpdated {
                        id: branch_id,
                        status: BranchStatus::Failed {
                            reason: d.lines().next().unwrap_or("failed").to_string(),
                        },
                        elapsed_ms: t0.elapsed().as_millis() as u64,
                    });
                    if d.contains("unsolved goals") {
                        unsolved_diags.push((script.clone(), d.clone()));
                    }
                    last_diag = d;
                }
            }
        }
        return Ok((None, last_diag, unsolved_diags));
    };

    let prefix = strip_trailing_docstrings_and_attrs(&src[..thm_pos]);
    let rest = &src[thm_pos..];
    let Some(by_rel) = rest.find(":= by") else {
        return Ok((None, last_diag, unsolved_diags));
    };
    let after_name = &rest[needle.len()..by_rel];
    let hb = max_heartbeats.max(10_000);

    const BATCH_SIZE: usize = 8;
    for (chunk_idx, chunk) in candidates.chunks(BATCH_SIZE).enumerate() {
        let chunk_base_id = branch_offset + chunk_idx * BATCH_SIZE;
        let batch_start = Instant::now();
        let mut batch_src = String::from(prefix);
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(chunk.len());
        let mut cur_line = batch_src.lines().count()
            + if batch_src.ends_with('\n') || batch_src.is_empty() {
                1
            } else {
                2
            };
        if !batch_src.ends_with('\n') && !batch_src.is_empty() {
            batch_src.push('\n');
        }

        for (i, script) in chunk.iter().enumerate() {
            let start_line = cur_line;
            let body = script
                .iter()
                .map(|l| format!("  {l}"))
                .collect::<Vec<_>>()
                .join("\n");
            let decl = format!(
                "set_option maxHeartbeats {hb} in\nprivate theorem __probe_{i}{after_name}:= by\n{body}\n"
            );
            cur_line += decl.lines().count();
            ranges.push((start_line, cur_line.saturating_sub(1)));
            batch_src.push_str(&decl);

            let branch_id = chunk_base_id + i + 1;
            let preview = script.join("; ");
            ui.send(UiEvent::BranchSpawned {
                id: branch_id,
                tag: "Batch".to_string(),
                preview,
            });
            ui.send(UiEvent::BranchUpdated {
                id: branch_id,
                status: BranchStatus::Running,
                elapsed_ms: 0,
            });
        }

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let tmp_file = std::env::temp_dir().join(format!(
            "sheaf_batch_{}_{unique}_{:?}.lean",
            thm.name,
            std::thread::current().id()
        ));
        std::fs::write(&tmp_file, &batch_src)
            .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))?;

        let mut child = spawn_lean_json_piped(&tmp_file, lake_root)?;
        let mut cand_errors: Vec<Option<String>> = vec![None; chunk.len()];
        let mut has_unmapped_error = false;

        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            for line_res in reader.lines() {
                let Ok(line) = line_res else {
                    continue;
                };
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    let sev = v.get("severity").and_then(|s| s.as_str()).unwrap_or("");
                    if sev != "error" {
                        continue;
                    }
                    let data = v
                        .get("data")
                        .and_then(|d| d.as_str())
                        .unwrap_or("")
                        .to_string();
                    let err_line = v
                        .get("pos")
                        .and_then(|p| p.get("line"))
                        .and_then(|l| l.as_u64())
                        .map(|l| l as usize);
                    if let Some(el) = err_line
                        && let Some(idx) = ranges.iter().position(|&(s, e)| el >= s && el <= e)
                    {
                        if cand_errors[idx].is_none() {
                            ui.send(UiEvent::BranchUpdated {
                                id: chunk_base_id + idx + 1,
                                status: BranchStatus::Failed {
                                    reason: data.lines().next().unwrap_or("error").to_string(),
                                },
                                elapsed_ms: batch_start.elapsed().as_millis() as u64,
                            });
                            cand_errors[idx] = Some(data);
                        }
                    } else {
                        has_unmapped_error = true;
                        last_diag = data;
                    }
                }
            }
        }
        let _ = child.wait();
        let _ = std::fs::remove_file(&tmp_file);

        if !has_unmapped_error {
            for (i, err) in cand_errors.iter().enumerate() {
                if err.is_none() {
                    let script = &chunk[i];
                    match audit_single_candidate(src, &thm.name, full_name, script, lake_root)? {
                        Ok(cand_src) => {
                            let elapsed_ms = batch_start.elapsed().as_millis() as u64;
                            ui.send(UiEvent::BranchUpdated {
                                id: chunk_base_id + i + 1,
                                status: BranchStatus::Verified {
                                    tactic: script.join("; "),
                                },
                                elapsed_ms,
                            });
                            for (j, other_err) in cand_errors.iter().enumerate() {
                                if j != i && other_err.is_none() {
                                    ui.send(UiEvent::BranchUpdated {
                                        id: chunk_base_id + j + 1,
                                        status: BranchStatus::Canceled,
                                        elapsed_ms,
                                    });
                                }
                            }
                            return Ok((
                                Some((cand_src, script.clone())),
                                String::new(),
                                unsolved_diags,
                            ));
                        }
                        Err(d) => {
                            ui.send(UiEvent::BranchUpdated {
                                id: chunk_base_id + i + 1,
                                status: BranchStatus::Failed {
                                    reason: d.lines().next().unwrap_or("audit failed").to_string(),
                                },
                                elapsed_ms: batch_start.elapsed().as_millis() as u64,
                            });
                            last_diag = d;
                        }
                    }
                } else if let Some(d) = err {
                    if d.contains("unsolved goals") {
                        unsolved_diags.push((chunk[i].clone(), d.clone()));
                    }
                    last_diag = d.clone();
                }
            }
        } else {
            // Parse-error fallback: re-run candidates individually and collect unsolved goals
            for (i, script) in chunk.iter().enumerate() {
                let branch_id = chunk_base_id + i + 1;
                match audit_single_candidate(src, &thm.name, full_name, script, lake_root)? {
                    Ok(cand_src) => {
                        let elapsed_ms = batch_start.elapsed().as_millis() as u64;
                        ui.send(UiEvent::BranchUpdated {
                            id: branch_id,
                            status: BranchStatus::Verified {
                                tactic: script.join("; "),
                            },
                            elapsed_ms,
                        });
                        return Ok((
                            Some((cand_src, script.clone())),
                            String::new(),
                            unsolved_diags,
                        ));
                    }
                    Err(d) => {
                        ui.send(UiEvent::BranchUpdated {
                            id: branch_id,
                            status: BranchStatus::Failed {
                                reason: d.lines().next().unwrap_or("failed").to_string(),
                            },
                            elapsed_ms: batch_start.elapsed().as_millis() as u64,
                        });
                        if d.contains("unsolved goals") {
                            unsolved_diags.push((script.clone(), d.clone()));
                        }
                        last_diag = d;
                    }
                }
            }
        }
    }

    Ok((None, last_diag, unsolved_diags))
}

/// Multi-file worktree candidate specification for concurrent evaluation.
#[derive(Clone, Debug)]
pub struct MultiFileCandidate {
    pub tree: TreeSnapshot,
    pub target_rel_path: PathBuf,
    pub theorem_full_name: String,
    pub script_label: Vec<String>,
    pub simulated_delay_ms: u64,
}

/// Evaluate multi-file `TreeSnapshot` branches concurrently in ephemeral `<temp>/sheaf_wt_<hex12>_*` worktrees
/// using a bounded worker pool (`max_workers`) and early `AtomicBool` cancellation.
pub fn evaluate_multifile_worktree_branches(
    store: &BlobStore,
    base_tree: &TreeSnapshot,
    candidates: &[MultiFileCandidate],
    max_workers: usize,
) -> Result<Option<(TreeSnapshot, Vec<String>)>, ProverError> {
    if candidates.is_empty() {
        return Ok(None);
    }

    let solved = Arc::new(AtomicBool::new(false));
    let worker_cap = max_workers.clamp(1, 8);
    let mut winner: Option<(TreeSnapshot, Vec<String>)> = None;

    for wave in candidates.chunks(worker_cap) {
        if solved.load(Ordering::Acquire) {
            break;
        }
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(wave.len());
            for cand in wave {
                let solved_flag = Arc::clone(&solved);
                handles.push(s.spawn(move || {
                    eval_single_multifile_worktree(store, base_tree, cand, &solved_flag)
                }));
            }
            for h in handles {
                if let Ok(Ok(Some(win))) = h.join()
                    && winner.is_none()
                {
                    winner = Some(win);
                }
            }
        });
    }

    Ok(winner)
}

fn eval_single_multifile_worktree(
    store: &BlobStore,
    base_tree: &TreeSnapshot,
    cand: &MultiFileCandidate,
    solved: &AtomicBool,
) -> Result<Option<(TreeSnapshot, Vec<String>)>, ProverError> {
    if solved.load(Ordering::Acquire) {
        return Ok(None);
    }

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let wt_dir = std::env::temp_dir().join(format!(
        "sheaf_wt_{}_{unique}_{:?}",
        cand.tree.tree_hash.hex12(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&wt_dir)
        .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))?;

    // Materialize dirty files (plus target file if not dirty) into ephemeral worktree
    let dirty = cand.tree.dirty_paths(base_tree);
    for rel_path in dirty {
        if let Some(content) = cand.tree.read_file(store, rel_path) {
            let full_path = wt_dir.join(rel_path);
            if let Some(parent) = full_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&full_path, content.as_ref());
        }
    }

    // Build combined translation unit from materialized files in dependency order
    // (imports at top, non-target helper files next, then target file + #print axioms)
    let mut imports = Vec::new();
    let mut body_src = String::new();
    for (rel_path, hash) in &cand.tree.files {
        if rel_path != &cand.target_rel_path
            && let Some(content) = store.get(hash)
        {
            for line in content.lines() {
                if line.trim_start().starts_with("import ") {
                    if !imports.contains(&line.to_string()) {
                        imports.push(line.to_string());
                    }
                } else {
                    body_src.push_str(line);
                    body_src.push('\n');
                }
            }
            body_src.push('\n');
        }
    }
    if let Some(target_content) = cand.tree.read_file(store, &cand.target_rel_path) {
        for line in target_content.lines() {
            if line.trim_start().starts_with("import ") {
                if !imports.contains(&line.to_string()) {
                    imports.push(line.to_string());
                }
            } else {
                body_src.push_str(line);
                body_src.push('\n');
            }
        }
    }
    let mut combined_src = imports.join("\n");
    if !combined_src.is_empty() {
        combined_src.push('\n');
    }
    combined_src.push_str(&body_src);
    combined_src.push_str(&format!("\n#print axioms {}\n", cand.theorem_full_name));

    let audit_file = wt_dir.join("__multifile_audit.lean");
    if let Err(e) = std::fs::write(&audit_file, &combined_src) {
        let _ = std::fs::remove_dir_all(&wt_dir);
        return Err(ProverError::Repl(lean_repl::ReplError::Io(e)));
    }

    // Optional delay for cancellation testing, checking `solved` every 5ms
    let mut slept = 0u64;
    while slept < cand.simulated_delay_ms {
        if solved.load(Ordering::Acquire) {
            let _ = std::fs::remove_dir_all(&wt_dir);
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(5));
        slept += 5;
    }

    if solved.load(Ordering::Acquire) {
        let _ = std::fs::remove_dir_all(&wt_dir);
        return Ok(None);
    }

    let mut child = match std::process::Command::new("lean")
        .arg("--json")
        .arg(&audit_file)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&wt_dir);
            return Err(ProverError::Repl(lean_repl::ReplError::Io(e)));
        }
    };

    loop {
        if solved.load(Ordering::Acquire) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_dir_all(&wt_dir);
            return Ok(None);
        }
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(e) => {
                let _ = child.kill();
                let _ = std::fs::remove_dir_all(&wt_dir);
                return Err(ProverError::Repl(lean_repl::ReplError::Io(e)));
            }
        }
    }

    let out = child
        .wait_with_output()
        .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))?;
    let _ = std::fs::remove_dir_all(&wt_dir);

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut has_error = !out.status.success();
    let mut axiom_ok = false;

    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let sev = v.get("severity").and_then(|s| s.as_str()).unwrap_or("");
            let data = v.get("data").and_then(|d| d.as_str()).unwrap_or("");
            if sev == "error" {
                has_error = true;
            }
            if data.contains("depends on axioms:")
                || data.contains("does not depend on any axioms")
            {
                if lean_repl::verify_allowed_axioms(data).is_ok() {
                    axiom_ok = true;
                } else {
                    has_error = true;
                }
            }
        }
    }

    if !has_error && axiom_ok {
        solved.store(true, Ordering::Release);
        Ok(Some((cand.tree.clone(), cand.script_label.clone())))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover_sorry_theorems;

    #[test]
    fn test_streaming_batch_probe_emits_live_branch_updates() {
        let dir = std::env::temp_dir().join("lean_prover_v17_streaming_probe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("StreamProbe.lean");
        let src = "import Init\ntheorem add_comm_stream (a b : Nat) : a + b = b + a := by\n  sorry\n";
        std::fs::write(&f, src).unwrap();
        let thms = discover_sorry_theorems(&f).unwrap();
        let cands = vec![
            vec!["rfl".to_string()],
            vec!["intro h".to_string()],
            vec!["omega".to_string()],
        ];
        let (ui, rx) = UiHandle::test_channel();
        let (solved, _diag, _unsolved) = verify_candidates_batched_with_heartbeats_and_ui(
            src,
            &thms[0],
            "add_comm_stream",
            &cands,
            None,
            200_000,
            &ui,
            0,
        )
        .unwrap();
        drop(ui);
        assert!(solved.is_some());
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(
            events.iter().any(|e| matches!(
                e,
                UiEvent::BranchUpdated {
                    id: 1,
                    status: BranchStatus::Failed { .. },
                    ..
                }
            )),
            "expected live Failed event for branch #1: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                UiEvent::BranchUpdated {
                    id: 3,
                    status: BranchStatus::Verified { .. },
                    ..
                }
            )),
            "expected Verified event for branch #3: {events:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
