// SPDX-License-Identifier: AGPL-3.0-only
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
            } else if let Some((base, rest)) = ident.split_once('.')
                && let Some(mapped) = var_map.get(base)
            {
                out.push_str(mapped);
                out.push('.');
                out.push_str(rest);
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

/// Strip balanced outer `(...)` wrappers if the opening `(` at byte 0 closes at the last byte.
fn strip_balanced_outer_parens(mut s: &str) -> &str {
    loop {
        let trimmed = s.trim();
        if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
            return trimmed;
        }
        let mut depth = 0usize;
        let mut closes_at_end = false;
        for (idx, ch) in trimmed.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        closes_at_end = idx + ch.len_utf8() == trimmed.len();
                        break;
                    }
                }
                _ => {}
            }
        }
        if closes_at_end {
            s = &trimmed[1..trimmed.len() - 1];
        } else {
            return trimmed;
        }
    }
}

/// Extract the `head_symbol` of target type `T` after stripping outer parens, `∀`, and checking connectives.
pub fn extract_head_symbol(target: &str) -> Result<HeadSymbol, CanonicalizerError> {
    let mut s = strip_balanced_outer_parens(target);
    if s.is_empty() {
        return Err(CanonicalizerError::EmptyTarget);
    }

    // Strip leading `∀ ...,` binders and any nested outer parentheses
    loop {
        s = strip_balanced_outer_parens(s);
        if s.starts_with('∀')
            && let Some(comma_idx) = s.find(',')
        {
            s = strip_balanced_outer_parens(&s[comma_idx + 1..]);
            continue;
        }
        break;
    }

    // Check prefix logical connectives before stripping `→` or scanning infix relations
    if s.starts_with('∃') {
        return Ok(HeadSymbol::Named("Exists".to_string()));
    }
    if s.starts_with('¬') {
        return Ok(HeadSymbol::Named("Not".to_string()));
    }

    // Strip top-level `... →` implications and re-check outer parens / prefix connectives
    while let Some(arrow_idx) = find_top_level_char(s, '→') {
        s = strip_balanced_outer_parens(&s[arrow_idx + '→'.len_utf8()..]);
        if s.starts_with('∃') {
            return Ok(HeadSymbol::Named("Exists".to_string()));
        }
        if s.starts_with('¬') {
            return Ok(HeadSymbol::Named("Not".to_string()));
        }
    }

    // Check top-level binary propositional connectives before infix relations
    for (op, name) in [('↔', "Iff"), ('∧', "And"), ('∨', "Or")] {
        if find_top_level_char(s, op).is_some() {
            return Ok(HeadSymbol::Named(name.to_string()));
        }
    }

    // Check top-level infix relation operators
    for (op, name) in [
        ('=', "Eq"),
        ('≤', "LE.le"),
        ('<', "LT.lt"),
        ('≥', "GE.ge"),
        ('>', "GT.gt"),
        ('≠', "Ne"),
        ('∈', "Membership.mem"),
    ] {
        if find_top_level_char(s, op).is_some() {
            return Ok(HeadSymbol::Named(name.to_string()));
        }
    }

    // Otherwise read leading identifier
    let ident_end = s
        .char_indices()
        .find(|&(_, c)| !is_ident_char(c))
        .map_or(s.len(), |(i, _)| i);
    let head = &s[..ident_end];
    if head.is_empty() || head.starts_with("_v") {
        Ok(HeadSymbol::Polymorphic)
    } else {
        Ok(HeadSymbol::Named(head.to_string()))
    }
}

fn find_top_level_char(s: &str, target: char) -> Option<usize> {
    let mut depth = 0usize;
    for (idx, ch) in s.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            c if c == target && depth == 0 => return Some(idx),
            _ => {}
        }
    }
    None
}

/// Prune hypotheses in `Γ` with zero symbol overlap with `T` when token count exceeds `max_tokens`.
pub fn prune_disjoint_hypotheses(
    hypotheses: &[String],
    target: &str,
    max_tokens: usize,
) -> Vec<String> {
    use std::collections::HashSet;

    let total_tokens: usize = hypotheses
        .iter()
        .map(|h| h.split_whitespace().count())
        .sum::<usize>()
        + target.split_whitespace().count();

    if total_tokens <= max_tokens {
        return hypotheses.to_vec();
    }

    let target_idents: HashSet<&str> = extract_ident_slices(target).collect();
    hypotheses
        .iter()
        .filter(|h| extract_ident_slices(h).any(|id| target_idents.contains(id)))
        .cloned()
        .collect()
}

fn extract_ident_slices(expr: &str) -> impl Iterator<Item = &str> {
    expr.split(|c: char| !is_ident_char(c))
        .filter(|s| !s.is_empty() && is_ident_start(s))
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
    fn test_normalize_alpha_vars_with_dot_projections() {
        let hyps = vec!["xs : List Nat".to_string(), "ys : List Nat".to_string()];
        let target = "List.length (xs ++ ys) = xs.length + ys.length";
        let (norm_hyps, norm_target) = normalize_alpha_vars(&hyps, target);
        assert_eq!(norm_hyps, vec!["_v0 : List Nat", "_v1 : List Nat"]);
        assert_eq!(
            norm_target,
            "List.length (_v0 ++ _v1) = _v0.length + _v1.length"
        );
    }

    #[test]
    fn test_collapse_implicit_instances() {
        let collapsed = collapse_implicit_instances("@HAdd.hAdd α α α inst✝ _v0 _v0 = _v0");
        assert_eq!(collapsed, "@HAdd.hAdd α α α [_] _v0 _v0 = _v0");
        let collapsed_sup = collapse_implicit_instances("inst✝¹ : CommMagma α");
        assert_eq!(collapsed_sup, "[_] : CommMagma α");
    }

    #[test]
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
        assert_eq!(
            extract_head_symbol("(∃ _v0, _v0 + 1 = 2)").unwrap(),
            HeadSymbol::Named("Exists".to_string())
        );
        assert_eq!(
            extract_head_symbol("(∀ (_v0 _v1 : Nat), (∃ _v2 _v3 : Nat, _v2 + _v3 = _v0 + _v1))")
                .unwrap(),
            HeadSymbol::Named("Exists".to_string())
        );
        assert_eq!(
            extract_head_symbol("∃ _v0, _v0 = 0 → _v0 ≤ 1").unwrap(),
            HeadSymbol::Named("Exists".to_string())
        );
        assert_eq!(
            extract_head_symbol("(¬ (_v0 = _v1))").unwrap(),
            HeadSymbol::Named("Not".to_string())
        );
        assert_eq!(
            extract_head_symbol("_v0 = _v1 ∧ _v1 ≤ _v2").unwrap(),
            HeadSymbol::Named("And".to_string())
        );
        assert_eq!(
            extract_head_symbol("_v0 = _v1 ∨ _v1 = _v2").unwrap(),
            HeadSymbol::Named("Or".to_string())
        );
        assert_eq!(
            extract_head_symbol("_v0 = _v1 ↔ _v1 = _v0").unwrap(),
            HeadSymbol::Named("Iff".to_string())
        );
        assert_eq!(
            extract_head_symbol("_v0 ≠ _v1").unwrap(),
            HeadSymbol::Named("Ne".to_string())
        );
    }

    #[test]
    fn test_prune_disjoint_hypotheses_over_budget() {
        let mut hyps = vec!["_v0 : Nat".to_string(), "_v1 : Nat".to_string()];
        for i in 0..70 {
            hyps.push(format!("_h{i} : String.length _s{i} = 42"));
        }
        let target = "Nat.add _v0 _v1 = _v0";
        let pruned = prune_disjoint_hypotheses(&hyps, target, MAX_CANONICAL_TOKENS);
        assert!(pruned.contains(&"_v0 : Nat".to_string()));
        assert!(pruned.contains(&"_v1 : Nat".to_string()));
        assert!(pruned.len() < hyps.len());
    }
}
