//! Zero-allocation Lean 4 expression canonicalizer and head-symbol extractor.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum token budget for `Lean-Sys1` input (`512 - head_max_len`).
pub const MAX_CANONICAL_TOKENS: usize = 320;

/// Extracted head symbol of a canonicalized Lean 4 target type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HeadSymbol {
    Named(String),
    Polymorphic,
}

impl HeadSymbol {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Named(name) => name.as_str(),
            Self::Polymorphic => "POLYMORPHIC",
        }
    }
}

/// Canonicalized representation of a Lean 4 proof goal (`Γ ⊢ T`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalGoal {
    pub hypotheses: Vec<String>,
    pub target: String,
    pub head_symbol: HeadSymbol,
}

#[derive(Debug, Error)]
pub enum CanonicalizerError {
    #[error("empty target expression")]
    EmptyTarget,
}

/// Replace bound and local variable names with positional tokens (`_v0`, `_v1`, ...).
pub fn normalize_alpha_vars(hypotheses: &[String], target: &str) -> (Vec<String>, String) {
    use std::collections::HashMap;

    let mut var_map: HashMap<String, String> = HashMap::new();

    // 1. Register local hypothesis names (e.g., "x : Nat" -> "x")
    for hyp in hypotheses {
        if let Some((lhs, _)) = hyp.split_once(':') {
            for name in lhs.split_whitespace() {
                if is_ident_start(name) && !var_map.contains_key(name) {
                    let idx = var_map.len();
                    var_map.insert(name.to_string(), format!("_v{idx}"));
                }
            }
        }
    }

    // 2. Register bound variables in quantifiers like "∀ (n m : Nat), ..."
    register_quantifier_binders(target, &mut var_map);

    let norm_hyps = hypotheses
        .iter()
        .map(|h| replace_idents(h, &var_map))
        .collect();
    let norm_target = replace_idents(target, &var_map);
    (norm_hyps, norm_target)
}

fn is_ident_start(s: &str) -> bool {
    s.chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '.' || c == '\'' || c == '✝'
}

fn register_quantifier_binders(expr: &str, var_map: &mut std::collections::HashMap<String, String>) {
    let mut rest = expr;
    while let Some(pos) = rest.find(['∀', '∃']) {
        let after = &rest[pos..];
        let mut chars = after.chars();
        chars.next(); // skip quantifier
        let tail = chars.as_str().trim_start();
        if let Some(after_paren) = tail.strip_prefix('(')
            && let Some(close) = after_paren.find(')')
        {
            let binder_group = &after_paren[..close];
            if let Some((lhs, _)) = binder_group.split_once(':') {
                for name in lhs.split_whitespace() {
                    if is_ident_start(name) && !var_map.contains_key(name) {
                        let idx = var_map.len();
                        var_map.insert(name.to_string(), format!("_v{idx}"));
                    }
                }
            }
            rest = &after_paren[close + 1..];
            continue;
        }
        rest = tail;
    }
}

fn replace_idents(expr: &str, var_map: &std::collections::HashMap<String, String>) -> String {
    let mut out = String::with_capacity(expr.len());
    let mut chars = expr.char_indices().peekable();
    while let Some(&(start, ch)) = chars.peek() {
        if ch.is_alphabetic() || ch == '_' {
            let mut end = start + ch.len_utf8();
            chars.next();
            while let Some(&(idx, next_ch)) = chars.peek() {
                if is_ident_char(next_ch) {
                    end = idx + next_ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            let ident = &expr[start..end];
            if let Some(mapped) = var_map.get(ident) {
                out.push_str(mapped);
            } else {
                out.push_str(ident);
            }
        } else {
            out.push(ch);
            chars.next();
        }
    }
    out
}

/// Collapse auto-generated hygienic typeclass instances (`inst✝`, `_inst_2`) to `[_]`.
pub fn collapse_implicit_instances(expr: &str) -> String {
    let mut out = String::with_capacity(expr.len());
    let mut chars = expr.char_indices().peekable();
    while let Some(&(start, ch)) = chars.peek() {
        if ch.is_alphabetic() || ch == '_' {
            let mut end = start + ch.len_utf8();
            chars.next();
            while let Some(&(idx, next_ch)) = chars.peek() {
                if is_ident_char(next_ch) || is_superscript_digit(next_ch) {
                    end = idx + next_ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            let ident = &expr[start..end];
            if is_hygienic_instance(ident) {
                out.push_str("[_]");
            } else {
                out.push_str(ident);
            }
        } else {
            out.push(ch);
            chars.next();
        }
    }
    out
}

fn is_superscript_digit(c: char) -> bool {
    matches!(c, '⁰' | '¹' | '²' | '³' | '⁴' | '⁵' | '⁶' | '⁷' | '⁸' | '⁹')
}

fn is_hygienic_instance(ident: &str) -> bool {
    ident.starts_with("inst✝") || ident.starts_with("_inst_")
}

/// Extract the `head_symbol` of target type `T` after stripping leading `∀` / `→`.
pub fn extract_head_symbol(_target: &str) -> Result<HeadSymbol, CanonicalizerError> {
    todo!("TASK-M0-003: implement target head_symbol extraction")
}

/// Prune hypotheses in `Γ` with zero symbol overlap with `T` when token count exceeds `max_tokens`.
pub fn prune_disjoint_hypotheses(
    _hypotheses: &[String],
    _target: &str,
    _max_tokens: usize,
) -> Vec<String> {
    todo!("TASK-M0-004: implement zero-overlap hypothesis pruning")
}

/// Full canonicalization pipeline for a Lean 4 goal.
pub fn canonicalize_goal(
    hypotheses: &[String],
    target: &str,
) -> Result<CanonicalGoal, CanonicalizerError> {
    let (norm_hyps, norm_target) = normalize_alpha_vars(hypotheses, target);
    let collapsed_hyps: Vec<String> = norm_hyps
        .iter()
        .map(|h| collapse_implicit_instances(h))
        .collect();
    let collapsed_target = collapse_implicit_instances(&norm_target);
    let head_symbol = extract_head_symbol(&collapsed_target)?;
    let pruned_hyps =
        prune_disjoint_hypotheses(&collapsed_hyps, &collapsed_target, MAX_CANONICAL_TOKENS);
    Ok(CanonicalGoal {
        hypotheses: pruned_hyps,
        target: collapsed_target,
        head_symbol,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_head_symbol_polymorphic_string() {
        assert_eq!(HeadSymbol::Polymorphic.as_str(), "POLYMORPHIC");
        assert_eq!(HeadSymbol::Named("Eq".to_string()).as_str(), "Eq");
        assert_eq!(MAX_CANONICAL_TOKENS, 320);
    }

    #[test]
    fn test_alpha_equivalence_normalization() {
        let hyps_a = vec!["x : Nat".to_string(), "y : Nat".to_string()];
        let target_a = "Nat.add x y = Nat.add y x";
        let hyps_b = vec!["a : Nat".to_string(), "b : Nat".to_string()];
        let target_b = "Nat.add a b = Nat.add b a";

        let (norm_ha, norm_ta) = normalize_alpha_vars(&hyps_a, target_a);
        let (norm_hb, norm_tb) = normalize_alpha_vars(&hyps_b, target_b);
        assert_eq!(norm_ha, norm_hb);
        assert_eq!(norm_ta, norm_tb);
        assert_eq!(norm_ha, vec!["_v0 : Nat", "_v1 : Nat"]);
        assert_eq!(norm_ta, "Nat.add _v0 _v1 = Nat.add _v1 _v0");

        let (_, quant_norm) = normalize_alpha_vars(&[], "∀ (n m : Nat), n + m = m + n");
        assert_eq!(quant_norm, "∀ (_v0 _v1 : Nat), _v0 + _v1 = _v1 + _v0");
    }

    #[test]
    fn test_collapse_implicit_instances() {
        let collapsed = collapse_implicit_instances("@HAdd.hAdd α α α inst✝ _v0 _v0 = _v0");
        assert_eq!(collapsed, "@HAdd.hAdd α α α [_] _v0 _v0 = _v0");
        let collapsed_sup = collapse_implicit_instances("inst✝¹ : CommMagma α");
        assert_eq!(collapsed_sup, "[_] : CommMagma α");
    }

    #[test]
    #[ignore = "TDD: TASK-M0-003"]
    fn test_extract_head_symbol() {
        assert_eq!(
            extract_head_symbol("Nat.add _v0 _v1 = Nat.add _v1 _v0").unwrap(),
            HeadSymbol::Named("Eq".to_string())
        );
        assert_eq!(
            extract_head_symbol("_v0 ≤ _v1").unwrap(),
            HeadSymbol::Named("LE.le".to_string())
        );
        assert_eq!(
            extract_head_symbol("∀ (_v0 : Nat), List.length _v1 = _v0").unwrap(),
            HeadSymbol::Named("Eq".to_string())
        );
        assert_eq!(
            extract_head_symbol("_v0 _v1").unwrap(),
            HeadSymbol::Polymorphic
        );
    }

    #[test]
    #[ignore = "TDD: TASK-M0-004"]
    fn test_prune_disjoint_hypotheses_over_budget() {
        let mut hyps = vec!["_v0 : Nat".to_string(), "_v1 : Nat".to_string()];
        for i in 0..40 {
            hyps.push(format!("_h{i} : String.length _s{i} = 42"));
        }
        let target = "Nat.add _v0 _v1 = _v0";
        let pruned = prune_disjoint_hypotheses(&hyps, target, MAX_CANONICAL_TOKENS);
        assert!(pruned.contains(&"_v0 : Nat".to_string()));
        assert!(pruned.contains(&"_v1 : Nat".to_string()));
        assert!(pruned.len() < hyps.len());
    }
}
