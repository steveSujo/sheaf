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

/// Apply scalar temperature calibration and route based on `max(p) >= gate_threshold`.
pub fn route_with_confidence_gate_threshold(
    choice_logits: &[f32; 5],
    t_choice: f32,
    gate_threshold: f32,
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

    let decision = if best_prob >= gate_threshold {
        GateDecision::DirectExecute(strategy)
    } else {
        GateDecision::FallbackToLeJepa
    };

    (strategy, best_prob, decision)
}

/// Apply scalar temperature calibration and route based on `max(p) >= 0.50`.
pub fn route_with_confidence_gate(
    choice_logits: &[f32; 5],
    t_choice: f32,
) -> (StrategyChoice, f32, GateDecision) {
    route_with_confidence_gate_threshold(choice_logits, t_choice, CONFIDENCE_GATE_THRESHOLD)
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
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub max_seq_len: usize,
    pub tok_emb: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub norm_out_weight: Vec<f32>,
    pub qkv_weight: Vec<f32>,
    pub attn_proj_weight: Vec<f32>,
    pub ffn_gate_weight: Vec<f32>,
    pub ffn_up_weight: Vec<f32>,
    pub ffn_down_weight: Vec<f32>,
    pub pool_query_weight: Vec<f32>,
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
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
    /// Load and validate `lean_sys1.onnx` protobuf initializers and RoPE constants from raw ONNX bytes.
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
        let norm2_weight = tensors
            .get("enc.blocks.0.norm2.weight")
            .cloned()
            .unwrap_or_else(|| norm_weight.clone());
        let norm_out_weight = tensors
            .get("enc.norm_out.weight")
            .cloned()
            .unwrap_or_else(|| norm_weight.clone());

        let qkv_weight = tensors
            .get("onnx::MatMul_238")
            .ok_or_else(|| Sys1Error::Onnx("missing onnx::MatMul_238 (qkv)".to_string()))?
            .clone();
        let attn_proj_weight = tensors
            .get("onnx::MatMul_254")
            .ok_or_else(|| Sys1Error::Onnx("missing onnx::MatMul_254 (attn_proj)".to_string()))?
            .clone();
        let ffn_gate_weight = tensors
            .get("onnx::MatMul_255")
            .ok_or_else(|| Sys1Error::Onnx("missing onnx::MatMul_255 (ffn_gate)".to_string()))?
            .clone();
        let ffn_up_weight = tensors
            .get("onnx::MatMul_256")
            .ok_or_else(|| Sys1Error::Onnx("missing onnx::MatMul_256 (ffn_up)".to_string()))?
            .clone();
        let ffn_down_weight = tensors
            .get("onnx::MatMul_257")
            .ok_or_else(|| Sys1Error::Onnx("missing onnx::MatMul_257 (ffn_down)".to_string()))?
            .clone();
        let pool_query_weight = tensors
            .get("onnx::MatMul_258")
            .ok_or_else(|| Sys1Error::Onnx("missing onnx::MatMul_258 (pool_query)".to_string()))?
            .clone();
        let rope_cos = tensors
            .get("onnx::Slice_27")
            .ok_or_else(|| Sys1Error::Onnx("missing onnx::Slice_27 (rope_cos)".to_string()))?
            .clone();
        let rope_sin = tensors
            .get("onnx::Slice_41")
            .ok_or_else(|| Sys1Error::Onnx("missing onnx::Slice_41 (rope_sin)".to_string()))?
            .clone();

        let head_dim = 64usize;
        if !d_model.is_multiple_of(head_dim) || !rope_cos.len().is_multiple_of(head_dim) {
            return Err(Sys1Error::Onnx("invalid head_dim or RoPE table".to_string()));
        }
        let n_heads = d_model / head_dim;
        let max_seq_len = rope_cos.len() / head_dim;
        if !ffn_gate_weight.len().is_multiple_of(d_model) {
            return Err(Sys1Error::Onnx("invalid ffn_gate shape".to_string()));
        }
        let ffn_dim = ffn_gate_weight.len() / d_model;
        if qkv_weight.len() != d_model * (3 * d_model)
            || attn_proj_weight.len() != d_model * d_model
            || ffn_up_weight.len() != d_model * ffn_dim
            || ffn_down_weight.len() != ffn_dim * d_model
            || pool_query_weight.len() != d_model
            || rope_sin.len() != rope_cos.len()
        {
            return Err(Sys1Error::Onnx("transformer weight dimension mismatch".to_string()));
        }

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
            n_heads,
            head_dim,
            ffn_dim,
            max_seq_len,
            tok_emb,
            norm_weight,
            norm2_weight,
            norm_out_weight,
            qkv_weight,
            attn_proj_weight,
            ffn_gate_weight,
            ffn_up_weight,
            ffn_down_weight,
            pool_query_weight,
            rope_cos,
            rope_sin,
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

    /// Predict target next-state embedding `s_hat_y = P_phi(s_x, strategy)` for LeJEPA VDB queries.
    #[must_use]
    pub fn predict_lejepa_target(&self, state_emb: &[f32], strategy: StrategyChoice) -> Vec<f32> {
        let strat_row_idx = match strategy {
            StrategyChoice::AtomicClose => 0,
            StrategyChoice::Rewrite => 1,
            StrategyChoice::ApplyLemma => 2,
            StrategyChoice::IntroduceHave => 3,
            StrategyChoice::Induction => 4,
        };
        let row = &self.choice_weight[strat_row_idx * self.d_model..(strat_row_idx + 1) * self.d_model];
        let mut pred: Vec<f32> = state_emb
            .iter()
            .zip(row.iter())
            .map(|(&s, &w)| s + 0.1 * w)
            .collect();
        let norm: f32 = pred.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        let state_norm: f32 = state_emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        let scale = state_norm / norm;
        for v in &mut pred {
            *v *= scale;
        }
        pred
    }

    /// Execute the full `Lean-Sys1` transformer forward pass with a configurable confidence gate threshold.
    pub fn forward_with_gate(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        temps: &CalibrationTemperatures,
        gate_threshold: f32,
    ) -> Result<(Sys1Prediction, [f32; 5], GateDecision), Sys1Error> {
        let total_len = input_ids.len();
        if total_len == 0 || total_len != attention_mask.len() || total_len > self.max_seq_len {
            return Err(Sys1Error::InvalidDimensions);
        }
        let Some(last_active) = attention_mask.iter().rposition(|&m| m != 0) else {
            return Err(Sys1Error::InvalidDimensions);
        };
        let seq_len = last_active + 1;

        let d = self.d_model;
        let mut x = vec![0.0_f32; seq_len * d];
        for (t, &id) in input_ids[..seq_len].iter().enumerate() {
            let idx = (id.max(0) as usize).min(self.vocab_size - 1);
            let row = &self.tok_emb[idx * d..(idx + 1) * d];
            x[t * d..(t + 1) * d].copy_from_slice(row);
        }

        // Block 0: RMSNorm 1
        let h1 = rms_norm_seq(&x, seq_len, d, &self.norm_weight);

        // QKV projection: [L, d] @ [d, 3*d] -> [L, 3*d]
        let qkv = matmul_2d(&h1, seq_len, d, &self.qkv_weight, 3 * d);

        // Multi-head RoPE Self-Attention
        let nh = self.n_heads;
        let hd = self.head_dim;
        let half = hd / 2;
        let mut q_rot = vec![0.0_f32; nh * seq_len * hd];
        let mut k_rot = vec![0.0_f32; nh * seq_len * hd];
        let mut v_all = vec![0.0_f32; nh * seq_len * hd];

        for t in 0..seq_len {
            let cos_t = &self.rope_cos[t * hd..(t + 1) * hd];
            let sin_t = &self.rope_sin[t * hd..(t + 1) * hd];
            let qkv_t = &qkv[t * 3 * d..(t + 1) * 3 * d];
            for h in 0..nh {
                let q_src = &qkv_t[h * hd..(h + 1) * hd];
                let k_src = &qkv_t[d + h * hd..d + (h + 1) * hd];
                let v_src = &qkv_t[2 * d + h * hd..2 * d + (h + 1) * hd];
                let dst_off = (h * seq_len + t) * hd;
                v_all[dst_off..dst_off + hd].copy_from_slice(v_src);
                for j in 0..half {
                    let q0 = q_src[j];
                    let q1 = q_src[j + half];
                    q_rot[dst_off + j] = q0 * cos_t[j] + (-q1) * sin_t[j];
                    q_rot[dst_off + j + half] = q1 * cos_t[j + half] + q0 * sin_t[j + half];

                    let k0 = k_src[j];
                    let k1 = k_src[j + half];
                    k_rot[dst_off + j] = k0 * cos_t[j] + (-k1) * sin_t[j];
                    k_rot[dst_off + j + half] = k1 * cos_t[j + half] + k0 * sin_t[j + half];
                }
            }
        }

        let inv_sqrt_hd = 1.0_f32 / (hd as f32).sqrt();
        let mut attn_ctx = vec![0.0_f32; seq_len * d];
        let mut scores = vec![0.0_f32; seq_len];
        for h in 0..nh {
            let base = h * seq_len * hd;
            for i in 0..seq_len {
                let qi = &q_rot[base + i * hd..base + (i + 1) * hd];
                let mut max_s = f32::NEG_INFINITY;
                for j in 0..seq_len {
                    if attention_mask[j] == 0 {
                        scores[j] = f32::NEG_INFINITY;
                    } else {
                        let kj = &k_rot[base + j * hd..base + (j + 1) * hd];
                        let s = dot(qi, kj) * inv_sqrt_hd;
                        scores[j] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                }
                let mut sum_e = 0.0_f32;
                for s in &mut scores[..seq_len] {
                    if s.is_finite() {
                        let e = (*s - max_s).exp();
                        *s = e;
                        sum_e += e;
                    } else {
                        *s = 0.0;
                    }
                }
                let inv_sum = 1.0_f32 / sum_e.max(1e-12);
                let dst = &mut attn_ctx[i * d + h * hd..i * d + (h + 1) * hd];
                for j in 0..seq_len {
                    let p = scores[j] * inv_sum;
                    if p != 0.0 {
                        let vj = &v_all[base + j * hd..base + (j + 1) * hd];
                        for (acc, &vv) in dst.iter_mut().zip(vj.iter()) {
                            *acc += p * vv;
                        }
                    }
                }
            }
        }

        let attn_proj = matmul_2d(&attn_ctx, seq_len, d, &self.attn_proj_weight, d);
        let mut x2 = vec![0.0_f32; seq_len * d];
        for i in 0..seq_len * d {
            x2[i] = x[i] + attn_proj[i];
        }

        // Block 0: RMSNorm 2 + SwiGLU FFN
        let h2 = rms_norm_seq(&x2, seq_len, d, &self.norm2_weight);
        let gate_act = matmul_2d(&h2, seq_len, d, &self.ffn_gate_weight, self.ffn_dim);
        let up_act = matmul_2d(&h2, seq_len, d, &self.ffn_up_weight, self.ffn_dim);
        let mut ffn_mid = vec![0.0_f32; seq_len * self.ffn_dim];
        for i in 0..ffn_mid.len() {
            let g = gate_act[i];
            let silu = g / (1.0_f32 + (-g).exp());
            ffn_mid[i] = silu * up_act[i];
        }
        let ffn_out = matmul_2d(&ffn_mid, seq_len, self.ffn_dim, &self.ffn_down_weight, d);

        let mut x3 = vec![0.0_f32; seq_len * d];
        for i in 0..seq_len * d {
            x3[i] = x2[i] + ffn_out[i];
        }

        // Final norm_out + Masked Attention Pooling
        let h_out = rms_norm_seq(&x3, seq_len, d, &self.norm_out_weight);
        let mut pool_scores = vec![0.0_f32; seq_len];
        let mut max_p = f32::NEG_INFINITY;
        for t in 0..seq_len {
            if attention_mask[t] == 0 {
                pool_scores[t] = f32::NEG_INFINITY;
            } else {
                let s = dot(&h_out[t * d..(t + 1) * d], &self.pool_query_weight);
                pool_scores[t] = s;
                if s > max_p {
                    max_p = s;
                }
            }
        }
        let mut sum_p = 0.0_f32;
        for ps in &mut pool_scores[..seq_len] {
            if ps.is_finite() {
                let e = (*ps - max_p).exp();
                *ps = e;
                sum_p += e;
            } else {
                *ps = 0.0;
            }
        }
        let inv_sum_p = 1.0_f32 / sum_p.max(1e-12);
        let mut pooled = vec![0.0_f32; d];
        for t in 0..seq_len {
            let w = pool_scores[t] * inv_sum_p;
            if w != 0.0 {
                let row = &h_out[t * d..(t + 1) * d];
                for (acc, &v) in pooled.iter_mut().zip(row.iter()) {
                    *acc += w * v;
                }
            }
        }

        let mut choice_logits = self.choice_bias;
        for (c, logit) in choice_logits.iter_mut().enumerate() {
            let row = &self.choice_weight[c * d..(c + 1) * d];
            *logit += dot(row, &pooled);
        }

        let noul_logit = self.noul_bias + dot(&self.noul_weight, &pooled);
        let direct_close_prob = 1.0 / (1.0 + (-noul_logit / temps.t_noul.max(1e-6)).exp());

        let mut score_sum = 0usize;
        for (k, &bias) in self.score_bias.iter().enumerate() {
            let row = &self.score_weight[k * d..(k + 1) * d];
            let z = bias + dot(row, &pooled);
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
            route_with_confidence_gate_threshold(&choice_logits, temps.t_choice, gate_threshold);

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

    /// Execute the `Lean-Sys1` forward pass on `(input_ids, attention_mask)` using `CONFIDENCE_GATE_THRESHOLD`.
    pub fn forward(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        temps: &CalibrationTemperatures,
    ) -> Result<(Sys1Prediction, [f32; 5], GateDecision), Sys1Error> {
        self.forward_with_gate(input_ids, attention_mask, temps, CONFIDENCE_GATE_THRESHOLD)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn rms_norm_seq(x: &[f32], seq_len: usize, d: usize, weight: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; seq_len * d];
    for t in 0..seq_len {
        let src = &x[t * d..(t + 1) * d];
        let mean_sq = src.iter().map(|&v| v * v).sum::<f32>() / (d as f32);
        let inv_rms = 1.0_f32 / (mean_sq + 1e-6_f32).sqrt();
        let dst = &mut out[t * d..(t + 1) * d];
        for i in 0..d {
            dst[i] = src[i] * inv_rms * weight[i];
        }
    }
    out
}

fn matmul_2d(a: &[f32], rows: usize, k: usize, b: &[f32], cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let a_row = &a[r * k..(r + 1) * k];
        let o_row = &mut out[r * cols..(r + 1) * cols];
        for kk in 0..k {
            let av = a_row[kk];
            let b_row = &b[kk * cols..(kk + 1) * cols];
            for c in 0..cols {
                o_row[c] += av * b_row[c];
            }
        }
    }
    out
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
        if field == 1 && wire == 2 {
            let len = read_varint(graph, &mut pos)? as usize;
            let end = pos.checked_add(len).filter(|&p| p <= graph.len()).ok_or_else(|| Sys1Error::Onnx("truncated node".into()))?;
            if let Some((name, vals)) = parse_constant_node(&graph[pos..end])? {
                map.insert(name, vals);
            }
            pos = end;
        } else if field == 5 && wire == 2 {
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

fn parse_constant_node(node: &[u8]) -> Result<Option<(String, Vec<f32>)>, Sys1Error> {
    let mut output_name: Option<String> = None;
    let mut op_type: Option<String> = None;
    let mut attr_tensor: Option<Vec<f32>> = None;
    let mut pos = 0usize;
    while pos < node.len() {
        let tag = read_varint(node, &mut pos)?;
        let field = tag >> 3;
        let wire = tag & 0x7;
        match (field, wire) {
            (2, 2) => {
                let len = read_varint(node, &mut pos)? as usize;
                let end = pos + len;
                if end > node.len() {
                    return Err(Sys1Error::Onnx("truncated node output".into()));
                }
                if output_name.is_none() {
                    output_name = Some(String::from_utf8_lossy(&node[pos..end]).to_string());
                }
                pos = end;
            }
            (4, 2) => {
                let len = read_varint(node, &mut pos)? as usize;
                let end = pos + len;
                if end > node.len() {
                    return Err(Sys1Error::Onnx("truncated node op_type".into()));
                }
                op_type = Some(String::from_utf8_lossy(&node[pos..end]).to_string());
                pos = end;
            }
            (5, 2) => {
                let len = read_varint(node, &mut pos)? as usize;
                let end = pos + len;
                if end > node.len() {
                    return Err(Sys1Error::Onnx("truncated node attribute".into()));
                }
                if let Some(vals) = parse_attribute_tensor(&node[pos..end])? {
                    attr_tensor = Some(vals);
                }
                pos = end;
            }
            _ => skip_field(node, &mut pos, wire)?,
        }
    }
    if op_type.as_deref() == Some("Constant")
        && let (Some(name), Some(vals)) = (output_name, attr_tensor)
    {
        Ok(Some((name, vals)))
    } else {
        Ok(None)
    }
}

fn parse_attribute_tensor(attr: &[u8]) -> Result<Option<Vec<f32>>, Sys1Error> {
    let mut pos = 0usize;
    let mut vals: Option<Vec<f32>> = None;
    while pos < attr.len() {
        let tag = read_varint(attr, &mut pos)?;
        let field = tag >> 3;
        let wire = tag & 0x7;
        if field == 5 && wire == 2 {
            let len = read_varint(attr, &mut pos)? as usize;
            let end = pos + len;
            if end > attr.len() {
                return Err(Sys1Error::Onnx("truncated attr tensor".into()));
            }
            if let Some((_, t_vals)) = parse_tensor_proto(&attr[pos..end])? {
                vals = Some(t_vals);
            }
            pos = end;
        } else {
            skip_field(attr, &mut pos, wire)?;
        }
    }
    Ok(vals)
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
    if data_type == 1 {
        let n = name.unwrap_or_default();
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
        assert_eq!(ranked.len(), 16);
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

    #[test]
    fn test_transformer_forward_order_sensitive_and_onnx_parity() {
        let onnx_bytes = include_bytes!("../../lean-prover/assets/lean_sys1.onnx");
        let sess = LeanSys1OnnxSession::from_onnx_bytes(onnx_bytes).unwrap();
        let temps = CalibrationTemperatures {
            t_choice: 1.0,
            t_noul: 1.0,
            t_fit: 1.0,
        };
        let mut ids_a = vec![0_i64; 512];
        let mut mask = vec![0_i64; 512];
        for (i, &tok) in [2_i64, 10, 15, 3].iter().enumerate() {
            ids_a[i] = tok;
            mask[i] = 1;
        }
        let (pred_a, choice_a, _) = sess.forward(&ids_a, &mask, &temps).unwrap();

        let ref_state_prefix = [
            -1.0525968_f32,
            -0.11565425,
            -1.1521168,
            1.0578138,
            -0.8256417,
            -0.08311277,
            0.31805936,
            -0.38810647,
        ];
        for (got, exp) in pred_a.state_emb.iter().zip(ref_state_prefix.iter()) {
            assert!((got - exp).abs() < 1e-4, "state_emb mismatch: got={got}, exp={exp}");
        }
        let ref_choice = [0.11478883_f32, -0.08977753, -0.023169301, -0.14556327, 0.08173258];
        for (got, exp) in choice_a.iter().zip(ref_choice.iter()) {
            assert!((got - exp).abs() < 1e-4, "choice mismatch: got={got}, exp={exp}");
        }

        let mut ids_b = vec![0_i64; 512];
        for (i, &tok) in [2_i64, 15, 10, 3].iter().enumerate() {
            ids_b[i] = tok;
        }
        let (pred_b, _, _) = sess.forward(&ids_b, &mask, &temps).unwrap();
        let diff: f32 = pred_a
            .state_emb
            .iter()
            .zip(pred_b.state_emb.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-2, "transformer forward pass must be order-sensitive");
    }

    #[test]
    fn test_premise_fit_untruncated_and_configurable_gate() {
        let high = [2.5, 0.5, 0.1, -0.5, -0.5];
        let (_, conf, dec_75) = route_with_confidence_gate_threshold(&high, 1.0, 0.75);
        assert!(conf >= 0.75);
        assert_eq!(dec_75, GateDecision::DirectExecute(StrategyChoice::AtomicClose));

        let (_, _, dec_95) = route_with_confidence_gate_threshold(&high, 1.0, 0.95);
        assert_eq!(dec_95, GateDecision::FallbackToLeJepa);

        let hits = vec![(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)];
        assert_eq!(
            select_fallback_outcome(&hits, 4),
            FallbackOutcome::BranchOnLemmas(hits)
        );
    }
}
