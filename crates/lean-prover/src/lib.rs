//! Best-First Prover Search Loop, Soft-Token Adapter, PDA Identifier Mask, and Prefix-Replay Reloader.

use std::collections::HashSet;
use thiserror::Error;

/// Standard Lean 4 tactic keywords always permitted by the PDA identifier mask.
pub const CORE_TACTIC_KEYWORDS: &[&str] = &[
    "rw", "apply", "exact", "have", "suffices", "simp", "omega", "aesop",
];

#[derive(Debug, Clone, PartialEq)]
pub struct SearchNode {
    pub state_id: u64,
    pub difficulty_bucket: u8,
    pub answer_confidence: f32,
    pub tactic_prefix: Vec<String>,
    pub retry_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairDecision {
    RetryWithDiagnostic(String),
    MarkBranchFailed,
}

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("search budget exhausted without proof")]
    SearchBudgetExhausted,
    #[error("REPL error: {0}")]
    Repl(#[from] lean_repl::ReplError),
}

/// Project `k <= 4` winning lemma embeddings through a 2-layer MLP into `k * m` soft tokens.
pub fn project_soft_tokens(
    lemma_embs: &[Vec<f32>],
    m_tokens: usize,
    d_gen: usize,
) -> Vec<Vec<f32>> {
    if lemma_embs.is_empty() || m_tokens == 0 || d_gen == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(lemma_embs.len().min(4) * m_tokens);
    for emb in lemma_embs.iter().take(4) {
        // 2-layer MLP projection: hidden = ReLU(W1 * emb), token_t = W2_t * hidden
        let d_in = emb.len();
        let hidden: Vec<f32> = (0..d_gen)
            .map(|h| {
                let acc: f32 = emb
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| v * (((h + j) % 7) as f32 * 0.05 - 0.15))
                    .sum();
                (acc / (d_in.max(1) as f32).sqrt()).max(0.0)
            })
            .collect();

        for t in 0..m_tokens {
            let scale = 1.0 / ((t + 1) as f32).sqrt();
            let token: Vec<f32> = hidden
                .iter()
                .enumerate()
                .map(|(i, &h)| h * scale * (1.0 + ((i + t) % 3) as f32 * 0.1))
                .collect();
            out.push(token);
        }
    }
    out
}

/// Build a dynamic PDA/trie logit mask allowing only hypotheses, top-k lemmas, and core tactics.
pub fn build_pda_logit_mask(
    vocab_tokens: &[String],
    hypotheses: &[String],
    top_k_lemmas: &[String],
    partial_prefix: &str,
) -> Vec<f32> {
    let mut allowed: HashSet<&str> = HashSet::new();
    for h in hypotheses {
        allowed.insert(h.as_str());
    }
    for l in top_k_lemmas {
        allowed.insert(l.as_str());
    }
    for &kw in CORE_TACTIC_KEYWORDS {
        allowed.insert(kw);
    }

    let ident_prefix = partial_prefix
        .rsplit([' ', '[', '(', ',', '\n', '\t'])
        .next()
        .unwrap_or("");

    vocab_tokens
        .iter()
        .map(|tok| {
            let is_allowed = allowed.contains(tok.as_str())
                && (ident_prefix.is_empty() || tok.starts_with(ident_prefix));
            if is_allowed {
                0.0_f32
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

/// Detect whether `required_module` is absent from `active_modules` and return the expanded module list.
pub fn plan_prefix_replay_modules(
    _active_modules: &[String],
    _required_module: &str,
) -> Option<Vec<String>> {
    todo!("TASK-M4-003: implement mid-proof missing module detection and prefix-replay plan")
}

/// Allow at most 1 error-conditioned repair retry per branch.
pub fn evaluate_error_repair(_retry_count: u8, _diagnostic: &str) -> RepairDecision {
    todo!("TASK-M4-004: implement single-retry error-conditioned diagnostic repair")
}

/// Run Best-First proof search with episodic memoization and zero-sorry axiom check.
pub fn prove_theorem_with_allowed_set(_allowed_lemmas: &HashSet<String>) -> Result<Vec<String>, ProverError> {
    todo!("TASK-M4-005: implement end-to-end Best-First proof search loop")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_core_tactic_keywords_present() {
        assert!(CORE_TACTIC_KEYWORDS.contains(&"omega"));
        assert!(CORE_TACTIC_KEYWORDS.contains(&"rw"));
        assert!(CORE_TACTIC_KEYWORDS.contains(&"aesop"));
    }

    #[test]
    fn test_soft_token_mlp_adapter_projection() {
        let embs = vec![vec![0.1_f32; 384], vec![0.2_f32; 384]];
        let soft = project_soft_tokens(&embs, 4, 384);
        assert_eq!(soft.len(), 8);
        assert_eq!(soft[0].len(), 384);
    }

    #[test]
    fn test_pda_trie_identifier_logit_mask_zero_hallucination() {
        let vocab = vec![
            "Nat.add_comm".to_string(),
            "Nat.add_assoc".to_string(),
            "Nat.mul_comm".to_string(),
        ];
        let mask = build_pda_logit_mask(
            &vocab,
            &["_v0".into(), "_v1".into()],
            &["Nat.add_comm".into(), "Nat.add_assoc".into()],
            "rw [Nat.add_",
        );
        assert_eq!(mask[0], 0.0);
        assert_eq!(mask[1], 0.0);
        assert!(mask[2].is_infinite() && mask[2].is_sign_negative());
    }

    #[test]
    #[ignore = "TDD: TASK-M4-003"]
    fn test_mid_proof_missing_module_prefix_replay() {
        let active = vec!["Init".to_string(), "Mathlib.Data.Nat.Basic".to_string()];
        assert!(plan_prefix_replay_modules(&active, "Init").is_none());
        let expanded = plan_prefix_replay_modules(&active, "Mathlib.Algebra.Ring.Basic").unwrap();
        assert!(expanded.contains(&"Mathlib.Algebra.Ring.Basic".to_string()));
    }

    #[test]
    #[ignore = "TDD: TASK-M4-004"]
    fn test_error_conditioned_single_retry_repair() {
        assert_eq!(
            evaluate_error_repair(0, "type mismatch"),
            RepairDecision::RetryWithDiagnostic("type mismatch".into())
        );
        assert_eq!(
            evaluate_error_repair(1, "type mismatch"),
            RepairDecision::MarkBranchFailed
        );
    }

    #[test]
    #[ignore = "TDD: TASK-M4-005"]
    fn test_best_first_end_to_end_mock_proof() {
        let allowed: HashSet<String> = ["Nat.add_comm".to_string()].into_iter().collect();
        let script = prove_theorem_with_allowed_set(&allowed).unwrap();
        assert_eq!(script, vec!["rw [Nat.add_comm]", "omega"]);
    }
}
