// SPDX-License-Identifier: AGPL-3.0-only
//! Pantograph JSON-RPC stdio client, metavariable coupling analyzer, and axiom verifier.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use thiserror::Error;

/// Single Lean 4 goal inside a Pantograph proof state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Goal {
    pub goal_id: String,
    pub target: String,
    pub head_symbol: String,
    pub hypotheses: Vec<String>,
    pub metavars: Vec<String>,
}

/// Pantograph proof state returned over JSON-RPC.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofState {
    pub state_id: u64,
    pub goals: Vec<Goal>,
}

/// Goal coupling partition for independent vs sequential search branching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalGroup {
    Independent(Goal),
    Sequential(Vec<Goal>),
}

#[derive(Debug, Error)]
pub enum ReplError {
    #[error("sorryAx detected in #print axioms output")]
    SorryAxiomDetected,
    #[error("JSON-RPC serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Pantograph tactic error: {0}")]
    TacticError(String),
}

/// Return `true` if two goals share at least one metavariable.
pub fn are_goals_coupled(a: &Goal, b: &Goal) -> bool {
    a.metavars.iter().any(|m| b.metavars.contains(m))
}

/// Partition goals into connected components: coupled goals are `Sequential`, isolated are `Independent`.
pub fn partition_coupled_goals(goals: &[Goal]) -> Vec<GoalGroup> {
    let n = goals.len();
    let mut visited = vec![false; n];
    let mut groups = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut component = Vec::new();
        visited[start] = true;

        while let Some(u) = stack.pop() {
            component.push(goals[u].clone());
            for v in 0..n {
                if !visited[v] && are_goals_coupled(&goals[u], &goals[v]) {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }

        if component.len() == 1 {
            groups.push(GoalGroup::Independent(component.into_iter().next().unwrap()));
        } else {
            groups.push(GoalGroup::Sequential(component));
        }
    }

    groups
}

/// Generic Pantograph JSON-RPC session over buffered stdio streams.
pub struct PantographSession<R: BufRead, W: Write> {
    pub reader: R,
    pub writer: W,
    pub loaded_modules: Vec<String>,
}

impl<R: BufRead, W: Write> PantographSession<R, W> {
    pub fn new(reader: R, writer: W, loaded_modules: Vec<String>) -> Self {
        Self {
            reader,
            writer,
            loaded_modules,
        }
    }

    /// Execute a tactic on `(state_id, goal_id)` and return the resulting `ProofState`.
    pub fn execute_tactic(
        &mut self,
        state_id: u64,
        goal_id: &str,
        tactic: &str,
    ) -> Result<ProofState, ReplError> {
        let req = serde_json::json!({
            "cmd": "goal.tactic",
            "payload": {
                "stateId": state_id,
                "goalId": goal_id,
                "tactic": tactic,
            }
        });
        serde_json::to_writer(&mut self.writer, &req)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;

        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        let state: ProofState = serde_json::from_str(line.trim())?;
        Ok(state)
    }
}

/// Parse `#print axioms` output and reject proofs containing `sorryAx`.
pub fn verify_zero_sorry(print_axioms_output: &str) -> Result<bool, ReplError> {
    if print_axioms_output.contains("sorryAx") {
        Err(ReplError::SorryAxiomDetected)
    } else {
        Ok(true)
    }
}

/// Result of running the real `lean --json` compiler kernel on a theorem + tactic script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeanKernelCheckResult {
    pub success: bool,
    pub has_unsolved_goals: bool,
    pub diagnostics: Vec<String>,
    pub axioms: Vec<String>,
    pub elapsed_ms: u128,
}

#[derive(Debug, Deserialize)]
struct LeanJsonDiagnostic {
    #[serde(default)]
    severity: String,
    #[serde(default)]
    data: String,
}

/// Verify a Lean 4 theorem and tactic script using the real `lean --json` compiler.
pub fn verify_theorem_with_lean4_kernel(
    theorem_name: &str,
    theorem_decl: &str,
    tactics: &[String],
) -> Result<LeanKernelCheckResult, ReplError> {
    use std::process::Command;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp_path = std::env::temp_dir().join(format!("lean_proof_check_{theorem_name}_{unique}.lean"));

    let mut source = format!("{theorem_decl} := by\n");
    for tac in tactics {
        source.push_str("  ");
        source.push_str(tac);
        source.push('\n');
    }
    source.push_str(&format!("#print axioms {theorem_name}\n"));
    std::fs::write(&tmp_path, source)?;

    let start = Instant::now();
    let output = Command::new("lean").arg("--json").arg(&tmp_path).output();
    let elapsed_ms = start.elapsed().as_millis();
    let _ = std::fs::remove_file(&tmp_path);

    let output = output?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut diagnostics = Vec::new();
    let mut axioms = Vec::new();
    let mut has_error = !output.status.success();
    let mut has_unsolved_goals = false;

    for line in stdout.lines().chain(stderr.lines()) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(diag) = serde_json::from_str::<LeanJsonDiagnostic>(trimmed) {
            if diag.severity == "error" {
                has_error = true;
            }
            if diag.data.contains("unsolved goals") {
                has_unsolved_goals = true;
            }
            if diag.data.contains("depends on axioms:") || diag.data.contains("does not depend on any axioms") {
                verify_zero_sorry(&diag.data)?;
                if let Some((_, rhs)) = diag.data.split_once("depends on axioms:") {
                    for ax in rhs.split([',', ' ']) {
                        let ax = ax.trim().trim_matches(|c| c == '[' || c == ']');
                        if !ax.is_empty() {
                            axioms.push(ax.to_string());
                        }
                    }
                }
            }
            diagnostics.push(diag.data);
        } else {
            diagnostics.push(trimmed.to_string());
        }
    }

    Ok(LeanKernelCheckResult {
        success: !has_error && !has_unsolved_goals,
        has_unsolved_goals,
        diagnostics,
        axioms,
        elapsed_ms,
    })
}

/// Standard Lean 4 axioms permitted in verified solutions.
pub const ALLOWED_LEAN_AXIOMS: &[&str] = &["propext", "Classical.choice", "Quot.sound"];

/// Verify `#print axioms` output contains no `sorryAx` or non-standard axioms.
pub fn verify_allowed_axioms(print_axioms_output: &str) -> Result<Vec<String>, ReplError> {
    verify_zero_sorry(print_axioms_output)?;
    let mut axioms = Vec::new();
    if let Some((_, rhs)) = print_axioms_output.split_once("depends on axioms:") {
        let inner = rhs.trim().trim_start_matches('[').trim_end_matches(']');
        for ax in inner.split(',') {
            let ax = ax.trim();
            if ax.is_empty() {
                continue;
            }
            if !ALLOWED_LEAN_AXIOMS.contains(&ax) {
                return Err(ReplError::TacticError(format!("illegal axiom: {ax}")));
            }
            axioms.push(ax.to_string());
        }
    }
    Ok(axioms)
}

fn extract_normalized_signature(lean_src: &str, short_name: &str) -> Result<String, ReplError> {
    let needle = format!("theorem {short_name}");
    let start = lean_src
        .find(&needle)
        .ok_or_else(|| ReplError::TacticError(format!("signature for {short_name} not found")))?;
    let rest = &lean_src[start..];
    let end = rest
        .find(":= by")
        .ok_or_else(|| ReplError::TacticError(format!("':= by' for {short_name} not found")))?;
    Ok(rest[..end].split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Run Gate 1 (Signature Lock) + Gate 2 & 3 (Lean 4 Kernel + Axiom Audit) on `solutions/<id>.lean`.
pub fn verify_solution_file_in_workspace(
    workspace_dir: &std::path::Path,
    problem_rel_path: &str,
    solution_rel_path: &str,
    theorem_full_name: &str,
) -> Result<LeanKernelCheckResult, ReplError> {
    use std::process::Command;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    let prob_src = std::fs::read_to_string(workspace_dir.join(problem_rel_path))?;
    let sol_src = std::fs::read_to_string(workspace_dir.join(solution_rel_path))?;
    let short_name = theorem_full_name.rsplit('.').next().unwrap_or(theorem_full_name);

    // Gate 1: Signature Lock
    let prob_sig = extract_normalized_signature(&prob_src, short_name)?;
    let sol_sig = extract_normalized_signature(&sol_src, short_name)?;
    if prob_sig != sol_sig {
        return Err(ReplError::TacticError(
            "Signature mismatch: theorem statement was modified".to_string(),
        ));
    }

    // Gate 2 & 3: Kernel Elaboration + Axiom Audit
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp_path = workspace_dir.join(format!("_audit_{short_name}_{unique}.lean"));
    std::fs::write(&tmp_path, format!("{sol_src}\n\n#print axioms {theorem_full_name}\n"))?;

    let start = Instant::now();
    let output = Command::new("lean").arg("--json").arg(&tmp_path).output();
    let elapsed_ms = start.elapsed().as_millis();
    let _ = std::fs::remove_file(&tmp_path);

    let output = output?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut diagnostics = Vec::new();
    let mut axioms = Vec::new();
    let mut has_error = !output.status.success();
    let mut has_unsolved_goals = false;

    for line in stdout.lines().chain(stderr.lines()) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(diag) = serde_json::from_str::<LeanJsonDiagnostic>(trimmed) {
            if diag.severity == "error" {
                has_error = true;
            }
            if diag.data.contains("unsolved goals") {
                has_unsolved_goals = true;
            }
            if diag.data.contains("depends on axioms:") || diag.data.contains("does not depend on any axioms") {
                axioms = verify_allowed_axioms(&diag.data)?;
            }
            diagnostics.push(diag.data);
        } else {
            diagnostics.push(trimmed.to_string());
        }
    }

    Ok(LeanKernelCheckResult {
        success: !has_error && !has_unsolved_goals,
        has_unsolved_goals,
        diagnostics,
        axioms,
        elapsed_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn contract_proof_state_schema_roundtrip() {
        let json = r#"{
          "state_id": 42,
          "goals": [{
            "goal_id": "g_0",
            "target": "Nat.add _v0 _v1 = Nat.add _v1 _v0",
            "head_symbol": "Eq",
            "hypotheses": ["_v0 : Nat", "_v1 : Nat"],
            "metavars": []
          }]
        }"#;
        let state: ProofState = serde_json::from_str(json).expect("valid ProofState JSON");
        assert_eq!(state.state_id, 42);
        assert_eq!(state.goals.len(), 1);
    }

    #[test]
    fn test_metavariable_coupling_partition() {
        let g0 = Goal {
            goal_id: "g_0".into(),
            target: "A".into(),
            head_symbol: "Eq".into(),
            hypotheses: vec![],
            metavars: vec!["?m1".into()],
        };
        let g1 = Goal {
            goal_id: "g_1".into(),
            target: "B".into(),
            head_symbol: "Eq".into(),
            hypotheses: vec![],
            metavars: vec!["?m1".into(), "?m2".into()],
        };
        let g2 = Goal {
            goal_id: "g_2".into(),
            target: "C".into(),
            head_symbol: "Eq".into(),
            hypotheses: vec![],
            metavars: vec![],
        };
        let g3 = Goal {
            goal_id: "g_3".into(),
            target: "D".into(),
            head_symbol: "Eq".into(),
            hypotheses: vec![],
            metavars: vec!["?m3".into()],
        };

        assert!(are_goals_coupled(&g0, &g1));
        assert!(!are_goals_coupled(&g0, &g2));
        let groups = partition_coupled_goals(&[g0, g1, g2, g3]);
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn test_pantograph_jsonrpc_state_deserialization_and_step() {
        let mock_response = b"{\"state_id\":43,\"goals\":[]}\n";
        let reader = Cursor::new(mock_response.as_slice());
        let writer: Vec<u8> = Vec::new();
        let mut session = PantographSession::new(reader, writer, vec!["Init".into()]);
        let next_state = session.execute_tactic(42, "g_0", "omega").unwrap();
        assert_eq!(next_state.state_id, 43);
        assert!(next_state.goals.is_empty());
    }

    #[test]
    fn test_verify_zero_sorry_axioms() {
        assert!(verify_zero_sorry("'thm' depends on axioms: [propext, Quot.sound]").unwrap());
        let err = verify_zero_sorry("'thm' depends on axioms: [propext, sorryAx]").unwrap_err();
        assert!(matches!(err, ReplError::SorryAxiomDetected));
    }
}
