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
        _state_id: u64,
        _goal_id: &str,
        _tactic: &str,
    ) -> Result<ProofState, ReplError> {
        todo!("TASK-M0-006: implement Pantograph JSON-RPC tactic execution")
    }
}

/// Parse `#print axioms` output and reject proofs containing `sorryAx`.
pub fn verify_zero_sorry(_print_axioms_output: &str) -> Result<bool, ReplError> {
    todo!("TASK-M0-007: implement zero-sorry axiom verification")
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
    #[ignore = "TDD: TASK-M0-006"]
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
    #[ignore = "TDD: TASK-M0-007"]
    fn test_verify_zero_sorry_axioms() {
        assert!(verify_zero_sorry("'thm' depends on axioms: [propext, Quot.sound]").unwrap());
        let err = verify_zero_sorry("'thm' depends on axioms: [propext, sorryAx]").unwrap_err();
        assert!(matches!(err, ReplError::SorryAxiomDetected));
    }
}
