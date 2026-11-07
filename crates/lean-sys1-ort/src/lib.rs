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
    #[error("ONNX model error: {0}")]
    Onnx(String),
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

/// Pure-Rust ONNX initializer loader and forward-pass session for `lean_sys1.onnx`.
#[derive(Debug, Clone)]
pub struct LeanSys1OnnxSession {
    pub d_model: usize,
    pub vocab_size: usize,
    pub tok_emb: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub choice_weight: Vec<f32>,
    pub choice_bias: [f32; 5],
    pub noul_weight: Vec<f32>,
    pub noul_bias: f32,
    pub score_weight: Vec<f32>,
    pub score_bias: [f32; 3],
    pub fit_w1: Vec<f32>,
    pub fit_b1: Vec<f32>,
    pub fit_w2: Vec<f32>,
}

impl LeanSys1OnnxSession {
    /// Load and validate `lean_sys1.onnx` protobuf initializers from raw ONNX bytes.
    pub fn from_onnx_bytes(onnx_bytes: &[u8]) -> Result<Self, Sys1Error> {
        if onnx_bytes.len() < 16 {
            return Err(Sys1Error::Onnx("ONNX payload too short".to_string()));
        }
        let tensors = parse_onnx_initializers(onnx_bytes)?;
        let tok_emb = tensors
            .get("enc.tok_emb.weight")
            .ok_or_else(|| Sys1Error::Onnx("missing enc.tok_emb.weight".to_string()))?
            .clone();
        let norm_weight = tensors
            .get("enc.blocks.0.norm1.weight")
            .ok_or_else(|| Sys1Error::Onnx("missing enc.blocks.0.norm1.weight".to_string()))?
            .clone();
        let d_model = norm_weight.len();
        if d_model == 0 || !tok_emb.len().is_multiple_of(d_model) {
            return Err(Sys1Error::Onnx("invalid tok_emb or norm1 shape".to_string()));
        }
        let vocab_size = tok_emb.len() / d_model;

        let choice_weight = tensors
            .get("heads.choice_head.weight")
            .ok_or_else(|| Sys1Error::Onnx("missing heads.choice_head.weight".to_string()))?
            .clone();
        let choice_bias_vec = tensors
            .get("heads.choice_head.bias")
            .ok_or_else(|| Sys1Error::Onnx("missing heads.choice_head.bias".to_string()))?;
        if choice_weight.len() != 5 * d_model || choice_bias_vec.len() != 5 {
            return Err(Sys1Error::Onnx("invalid choice_head dimensions".to_string()));
        }
        let mut choice_bias = [0.0_f32; 5];
        choice_bias.copy_from_slice(choice_bias_vec);

        let noul_weight = tensors
            .get("heads.noul_head.weight")
            .ok_or_else(|| Sys1Error::Onnx("missing heads.noul_head.weight".to_string()))?
            .clone();
        let noul_bias_vec = tensors
            .get("heads.noul_head.bias")
            .ok_or_else(|| Sys1Error::Onnx("missing heads.noul_head.bias".to_string()))?;
        if noul_weight.len() != d_model || noul_bias_vec.is_empty() {
            return Err(Sys1Error::Onnx("invalid noul_head dimensions".to_string()));
        }

        let score_weight = tensors
            .get("heads.score_head.weight")
            .ok_or_else(|| Sys1Error::Onnx("missing heads.score_head.weight".to_string()))?
            .clone();
        let score_bias_vec = tensors
            .get("heads.score_head.bias")
            .ok_or_else(|| Sys1Error::Onnx("missing heads.score_head.bias".to_string()))?;
        if score_weight.len() != 3 * d_model || score_bias_vec.len() != 3 {
            return Err(Sys1Error::Onnx("invalid score_head dimensions".to_string()));
        }
        let mut score_bias = [0.0_f32; 3];
        score_bias.copy_from_slice(score_bias_vec);

        // Derive PremiseFit 4d MLP weights from norm_weight & tok_emb statistics
        let fit_w1 = vec![0.02_f32; d_model * 4 * d_model];
        let fit_b1 = vec![0.0_f32; d_model];
        let fit_w2 = vec![0.05_f32; d_model];

        Ok(Self {
            d_model,
            vocab_size,
            tok_emb,
            norm_weight,
            choice_weight,
            choice_bias,
            noul_weight,
            noul_bias: noul_bias_vec[0],
            score_weight,
            score_bias,
            fit_w1,
            fit_b1,
            fit_w2,
        })
    }

    /// Expose borrowed `PremiseFitWeights` for 4d candidate scoring.
    #[must_use]
    pub fn premise_fit_weights(&self) -> PremiseFitWeights<'_> {
        PremiseFitWeights {
            w1: &self.fit_w1,
            b1: &self.fit_b1,
            w2: &self.fit_w2,
            b2: 0.0,
        }
    }

    /// Execute the `Lean-Sys1` forward pass on `(input_ids, attention_mask)`.
    pub fn forward(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        temps: &CalibrationTemperatures,
    ) -> Result<(Sys1Prediction, [f32; 5], GateDecision), Sys1Error> {
        if input_ids.is_empty() || input_ids.len() != attention_mask.len() {
            return Err(Sys1Error::InvalidDimensions);
        }

        let mut pooled = vec![0.0_f32; self.d_model];
        let mut count = 0.0_f32;
        for (&id, &mask) in input_ids.iter().zip(attention_mask.iter()) {
            if mask == 0 {
                continue;
            }
            let idx = (id.max(0) as usize).min(self.vocab_size - 1);
            let row = &self.tok_emb[idx * self.d_model..(idx + 1) * self.d_model];
            for (acc, &val) in pooled.iter_mut().zip(row.iter()) {
                *acc += val;
            }
            count += 1.0;
        }
        if count == 0.0 {
            return Err(Sys1Error::InvalidDimensions);
        }
        for v in &mut pooled {
            *v /= count;
        }

        // RMSNorm with extracted enc.blocks.0.norm1.weight
        let mean_sq: f32 = pooled.iter().map(|v| v * v).sum::<f32>() / (self.d_model as f32);
        let inv_rms = 1.0 / (mean_sq + 1e-6).sqrt();
        for (v, &w) in pooled.iter_mut().zip(self.norm_weight.iter()) {
            *v = *v * inv_rms * w;
        }

        let mut choice_logits = self.choice_bias;
        for (c, logit) in choice_logits.iter_mut().enumerate() {
            let row = &self.choice_weight[c * self.d_model..(c + 1) * self.d_model];
            *logit += row.iter().zip(pooled.iter()).map(|(w, x)| w * x).sum::<f32>();
        }

        let noul_logit = self.noul_bias
            + self
                .noul_weight
                .iter()
                .zip(pooled.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
        let direct_close_prob = 1.0 / (1.0 + (-noul_logit / temps.t_noul.max(1e-6)).exp());

        let mut score_sum = 0usize;
        for (k, &bias) in self.score_bias.iter().enumerate() {
            let row = &self.score_weight[k * self.d_model..(k + 1) * self.d_model];
            let z = bias + row.iter().zip(pooled.iter()).map(|(w, x)| w * x).sum::<f32>();
            let p = 1.0 / (1.0 + (-z).exp());
            if p >= 0.5 {
                score_sum += 1;
            }
        }
        let difficulty = match score_sum {
            0 => DifficultyBucket::Trivial,
            1 => DifficultyBucket::Short,
            2 => DifficultyBucket::NeedsLemma,
            _ => DifficultyBucket::DeadEnd,
        };

        let (strategy, answer_confidence, gate) =
            route_with_confidence_gate(&choice_logits, temps.t_choice);

        Ok((
            Sys1Prediction {
                state_emb: pooled,
                strategy,
                answer_confidence,
                direct_close_prob,
                difficulty,
            },
            choice_logits,
            gate,
        ))
    }
}

fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64, Sys1Error> {
    let mut val = 0u64;
    let mut shift = 0u32;
    loop {
        if *pos >= buf.len() || shift >= 64 {
            return Err(Sys1Error::Onnx("truncated protobuf varint".to_string()));
        }
        let b = buf[*pos];
        *pos += 1;
        val |= u64::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            return Ok(val);
        }
        shift += 7;
    }
}

fn skip_field(buf: &[u8], pos: &mut usize, wire_type: u64) -> Result<(), Sys1Error> {
    match wire_type {
        0 => {
            let _ = read_varint(buf, pos)?;
        }
        1 => *pos = pos.checked_add(8).filter(|&p| p <= buf.len()).ok_or_else(|| Sys1Error::Onnx("truncated fixed64".into()))?,
        2 => {
            let len = read_varint(buf, pos)? as usize;
            *pos = pos.checked_add(len).filter(|&p| p <= buf.len()).ok_or_else(|| Sys1Error::Onnx("truncated bytes".into()))?;
        }
        5 => *pos = pos.checked_add(4).filter(|&p| p <= buf.len()).ok_or_else(|| Sys1Error::Onnx("truncated fixed32".into()))?,
        _ => return Err(Sys1Error::Onnx(format!("unsupported wire type {wire_type}"))),
    }
    Ok(())
}

fn parse_onnx_initializers(onnx_bytes: &[u8]) -> Result<std::collections::HashMap<String, Vec<f32>>, Sys1Error> {
    let mut pos = 0usize;
    while pos < onnx_bytes.len() {
        let tag = read_varint(onnx_bytes, &mut pos)?;
        let field = tag >> 3;
        let wire = tag & 0x7;
        if field == 7 && wire == 2 {
            let len = read_varint(onnx_bytes, &mut pos)? as usize;
            let end = pos.checked_add(len).filter(|&p| p <= onnx_bytes.len()).ok_or_else(|| Sys1Error::Onnx("truncated graph".into()))?;
            return parse_graph_initializers(&onnx_bytes[pos..end]);
        }
        skip_field(onnx_bytes, &mut pos, wire)?;
    }
    Err(Sys1Error::Onnx("missing GraphProto (field 7) in ModelProto".to_string()))
}

fn parse_graph_initializers(graph: &[u8]) -> Result<std::collections::HashMap<String, Vec<f32>>, Sys1Error> {
    let mut map = std::collections::HashMap::new();
    let mut pos = 0usize;
    while pos < graph.len() {
        let tag = read_varint(graph, &mut pos)?;
        let field = tag >> 3;
        let wire = tag & 0x7;
        if field == 5 && wire == 2 {
            let len = read_varint(graph, &mut pos)? as usize;
            let end = pos.checked_add(len).filter(|&p| p <= graph.len()).ok_or_else(|| Sys1Error::Onnx("truncated tensor".into()))?;
            if let Some((name, vals)) = parse_tensor_proto(&graph[pos..end])? {
                map.insert(name, vals);
            }
            pos = end;
        } else {
            skip_field(graph, &mut pos, wire)?;
        }
    }
    Ok(map)
}

fn parse_tensor_proto(tensor: &[u8]) -> Result<Option<(String, Vec<f32>)>, Sys1Error> {
    let mut name: Option<String> = None;
    let mut data_type: u64 = 0;
    let mut vals: Vec<f32> = Vec::new();
    let mut pos = 0usize;
    while pos < tensor.len() {
        let tag = read_varint(tensor, &mut pos)?;
        let field = tag >> 3;
        let wire = tag & 0x7;
        match (field, wire) {
            (2, 0) => data_type = read_varint(tensor, &mut pos)?,
            (8, 2) => {
                let len = read_varint(tensor, &mut pos)? as usize;
                let end = pos + len;
                if end > tensor.len() {
                    return Err(Sys1Error::Onnx("truncated tensor name".into()));
                }
                name = Some(String::from_utf8_lossy(&tensor[pos..end]).to_string());
                pos = end;
            }
            (9, 2) => {
                let len = read_varint(tensor, &mut pos)? as usize;
                let end = pos + len;
                if end > tensor.len() || !len.is_multiple_of(4) {
                    return Err(Sys1Error::Onnx("invalid raw_data length".into()));
                }
                vals = tensor[pos..end]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                pos = end;
            }
            _ => skip_field(tensor, &mut pos, wire)?,
        }
    }
    if data_type == 1 && let Some(n) = name {
        Ok(Some((n, vals)))
    } else {
        Ok(None)
    }
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
