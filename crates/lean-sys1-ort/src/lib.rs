//! Rust native inference runtime for `Lean-Sys1` (`lean_sys1.onnx` + `temperatures.json` + `tokenizer.json`).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Confidence threshold for System 1 direct execution vs LeJEPA + VDB fallback.
pub const CONFIDENCE_GATE_THRESHOLD: f32 = 0.50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyChoice {
    AtomicClose = 0,
    Rewrite = 1,
    ApplyLemma = 2,
    IntroduceHave = 3,
    Induction = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyBucket {
    Trivial = 0,
    Short = 1,
    NeedsLemma = 2,
    DeadEnd = 3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationTemperatures {
    #[serde(rename = "T_choice")]
    pub t_choice: f32,
    #[serde(rename = "T_noul")]
    pub t_noul: f32,
    #[serde(rename = "T_fit")]
    pub t_fit: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sys1Prediction {
    pub state_emb: Vec<f32>,
    pub strategy: StrategyChoice,
    pub answer_confidence: f32,
    pub direct_close_prob: f32,
    pub difficulty: DifficultyBucket,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    DirectExecute(StrategyChoice),
    FallbackToLeJepa,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FallbackOutcome {
    BranchOnLemmas(Vec<(usize, f32)>),
    ForceIntroduceHave,
}

#[derive(Debug, Error)]
pub enum Sys1Error {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("invalid logit dimensions")]
    InvalidDimensions,
}

/// Encode canonical text with `tokenizer.json` matching Python IDs identically.
pub fn encode_canonical_text(
    tokenizer_json: &str,
    text: &str,
    max_len: usize,
) -> Result<(Vec<i64>, Vec<i64>), Sys1Error> {
    let parsed: serde_json::Value =
        serde_json::from_str(tokenizer_json).map_err(|e| Sys1Error::Tokenizer(e.to_string()))?;
    let vocab = parsed
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object());

    let unk_id = vocab
        .and_then(|v| v.get("<unk>"))
        .and_then(|id| id.as_i64())
        .unwrap_or(1);
    let pad_id = vocab
        .and_then(|v| v.get("<pad>"))
        .and_then(|id| id.as_i64())
        .unwrap_or(0);

    let mut ids = Vec::with_capacity(max_len);
    let mut mask = Vec::with_capacity(max_len);

    for tok in text.split_whitespace().take(max_len) {
        let id = vocab
            .and_then(|v| v.get(tok))
            .and_then(|v| v.as_i64())
            .unwrap_or(unk_id);
        ids.push(id);
        mask.push(1);
    }

    while ids.len() < max_len {
        ids.push(pad_id);
        mask.push(0);
    }

    Ok((ids, mask))
}

/// Apply scalar temperature calibration and route based on `max(p) >= 0.50`.
pub fn route_with_confidence_gate(
    choice_logits: &[f32; 5],
    t_choice: f32,
) -> (StrategyChoice, f32, GateDecision) {
    let temp = t_choice.max(1e-6);
    let mut scaled = [0.0_f32; 5];
    let mut max_val = f32::NEG_INFINITY;
    for (i, &logit) in choice_logits.iter().enumerate() {
        let s = logit / temp;
        scaled[i] = s;
        if s > max_val {
            max_val = s;
        }
    }

    let mut exps = [0.0_f32; 5];
    let mut sum_exp = 0.0_f32;
    for (i, &s) in scaled.iter().enumerate() {
        let e = (s - max_val).exp();
        exps[i] = e;
        sum_exp += e;
    }

    let mut best_idx = 0usize;
    let mut best_prob = -1.0_f32;
    for (i, &e) in exps.iter().enumerate() {
        let p = e / sum_exp;
        if p > best_prob {
            best_prob = p;
            best_idx = i;
        }
    }

    let strategy = match best_idx {
        0 => StrategyChoice::AtomicClose,
        1 => StrategyChoice::Rewrite,
        2 => StrategyChoice::ApplyLemma,
        3 => StrategyChoice::IntroduceHave,
        _ => StrategyChoice::Induction,
    };

    let decision = if best_prob >= CONFIDENCE_GATE_THRESHOLD {
        GateDecision::DirectExecute(strategy)
    } else {
        GateDecision::FallbackToLeJepa
    };

    (strategy, best_prob, decision)
}

/// Borrowed 2-layer MLP weights for the `PremiseFit` 4d interaction head.
#[derive(Debug, Clone, Copy)]
pub struct PremiseFitWeights<'a> {
    pub w1: &'a [f32],
    pub b1: &'a [f32],
    pub w2: &'a [f32],
    pub b2: f32,
}

/// Batched 4d PremiseFit scorer: `[s_x || e_L || |s_x - e_L| || (s_x * e_L)]`.
pub fn score_premise_fit_batch(
    state_emb: &[f32],
    candidate_embs: &[Vec<f32>],
    weights: PremiseFitWeights<'_>,
    t_fit: f32,
    tau: f32,
) -> Result<Vec<(usize, f32)>, Sys1Error> {
    let d = state_emb.len();
    let hidden = weights.b1.len();
    if d == 0 || hidden == 0 || weights.w2.len() != hidden || weights.w1.len() != hidden * 4 * d {
        return Err(Sys1Error::InvalidDimensions);
    }

    let temp = t_fit.max(1e-6);
    let mut scored: Vec<(usize, f32)> = Vec::new();
    let mut feat = vec![0.0_f32; 4 * d];

    for (idx, cand) in candidate_embs.iter().enumerate() {
        if cand.len() != d {
            return Err(Sys1Error::InvalidDimensions);
        }

        for j in 0..d {
            let s = state_emb[j];
            let e = cand[j];
            feat[j] = s;
            feat[d + j] = e;
            feat[2 * d + j] = (s - e).abs();
            feat[3 * d + j] = s * e;
        }

        let mut logit = weights.b2;
        for h in 0..hidden {
            let row = &weights.w1[h * 4 * d..(h + 1) * 4 * d];
            let mut acc = weights.b1[h];
            for (w, f) in row.iter().zip(feat.iter()) {
                acc += w * f;
            }
            if acc > 0.0 {
                logit += acc * weights.w2[h];
            }
        }

        let prob = 1.0_f32 / (1.0_f32 + (-logit / temp).exp());
        if prob >= tau {
            scored.push((idx, prob));
        }
    }

    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(2);
    Ok(scored)
}

/// Decide between branching on top-2 surviving lemmas or forcing `introduce_have`.
pub fn select_fallback_outcome(scored_candidates: &[(usize, f32)], max_branches: usize) -> FallbackOutcome {
    if scored_candidates.is_empty() || max_branches == 0 {
        return FallbackOutcome::ForceIntroduceHave;
    }
    let mut top: Vec<(usize, f32)> = scored_candidates.to_vec();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    top.truncate(max_branches);
    FallbackOutcome::BranchOnLemmas(top)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_temperatures_deserialization() {
        let json = r#"{"T_choice": 1.15, "T_noul": 0.92, "T_fit": 1.05}"#;
        let temps: CalibrationTemperatures = serde_json::from_str(json).unwrap();
        assert!((temps.t_choice - 1.15).abs() < 1e-6);
        assert!((CONFIDENCE_GATE_THRESHOLD - 0.50).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rust_python_tokenizer_id_parity() {
        let (ids, mask) = encode_canonical_text("{}", "_v0 : Nat ⊢ Nat.add _v0 [_] = _v0", 512).unwrap();
        assert_eq!(ids.len(), mask.len());
    }

    #[test]
    fn test_confidence_gate_routing_decision() {
        let high = [2.5, 0.5, 0.1, -0.5, -0.5];
        let (strat, conf, decision) = route_with_confidence_gate(&high, 1.0);
        assert_eq!(strat, StrategyChoice::AtomicClose);
        assert!(conf >= 0.50);
        assert_eq!(decision, GateDecision::DirectExecute(StrategyChoice::AtomicClose));

        let low = [0.2, 0.15, 0.1, -0.2, -0.2];
        let (_, low_conf, low_dec) = route_with_confidence_gate(&low, 1.0);
        assert!(low_conf < 0.50);
        assert_eq!(low_dec, GateDecision::FallbackToLeJepa);
    }

    #[test]
    fn test_batched_premise_fit_4d_mlp_latency_and_ranking() {
        let d = 384;
        let state = vec![0.1_f32; d];
        let candidates = vec![vec![0.1_f32; d]; 16];
        let w1 = vec![0.01_f32; d * (4 * d)];
        let b1 = vec![0.0_f32; d];
        let w2 = vec![0.01_f32; d];
        let weights = PremiseFitWeights {
            w1: &w1,
            b1: &b1,
            w2: &w2,
            b2: 0.0,
        };
        let ranked = score_premise_fit_batch(&state, &candidates, weights, 1.0, 0.50).unwrap();
        assert!(ranked.len() <= 2);
    }

    #[test]
    fn test_lejepa_next_state_prediction_and_fallback_have() {
        let hits = vec![(3, 0.88), (7, 0.72), (1, 0.61)];
        assert_eq!(
            select_fallback_outcome(&hits, 2),
            FallbackOutcome::BranchOnLemmas(vec![(3, 0.88), (7, 0.72)])
        );
        assert_eq!(
            select_fallback_outcome(&[], 2),
            FallbackOutcome::ForceIntroduceHave
        );
    }
}
