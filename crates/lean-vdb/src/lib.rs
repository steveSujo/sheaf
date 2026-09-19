//! Embedded Vector DB (HNSW + SQ8), Two-Stage Dynamic Import Resolver, and Episodic Subgoal Cache.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Cosine similarity threshold above which a visited subgoal is pruned as a loop.
pub const EPISODIC_LOOP_THRESHOLD: f32 = 0.98;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LemmaMetadata {
    pub name: String,
    pub head_symbol: String,
    pub module: String,
    pub arity: u32,
    pub signature: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VdbSearchHit {
    pub metadata: LemmaMetadata,
    pub distance: f32,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Error)]
pub enum VdbError {
    #[error("invalid LVDB binary header")]
    InvalidHeader,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Two-stage dynamic import resolver (`symbol_to_module` + `import_dag`).
#[derive(Debug, Clone, Default)]
pub struct ImportResolver {
    pub symbol_to_module: HashMap<String, String>,
    pub import_dag: HashMap<String, Vec<String>>,
}

impl ImportResolver {
    /// Extract identifiers from theorem text and resolve defining modules.
    pub fn resolve_syntactic_modules(&self, theorem_stmt: &str) -> HashSet<String> {
        let mut modules = HashSet::new();
        for token in theorem_stmt.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')) {
            if let Some(module) = self.symbol_to_module.get(token) {
                modules.insert(module.clone());
            }
        }
        modules
    }

    /// Collapse candidate modules through `import_dag` to remove transitively implied modules.
    pub fn reduce_import_dag(&self, candidates: &HashSet<String>) -> Vec<String> {
        let mut implied = HashSet::new();
        for root in candidates {
            let mut stack: Vec<&str> = self
                .import_dag
                .get(root)
                .map(|deps| deps.iter().map(String::as_str).collect())
                .unwrap_or_default();
            while let Some(dep) = stack.pop() {
                if implied.insert(dep.to_string())
                    && let Some(next_deps) = self.import_dag.get(dep)
                {
                    stack.extend(next_deps.iter().map(String::as_str));
                }
            }
        }

        let mut reduced: Vec<String> = candidates
            .iter()
            .filter(|m| !implied.contains(*m))
            .cloned()
            .collect();
        reduced.sort();
        reduced
    }

    /// Combine syntactic closure, VDB top-16 modules, and core tactic pack, then reduce via DAG.
    pub fn resolve_preflight_modules(
        &self,
        theorem_stmt: &str,
        vdb_hit_modules: &[String],
        core_tactic_pack: &[String],
    ) -> Vec<String> {
        let mut union = self.resolve_syntactic_modules(theorem_stmt);
        union.extend(vdb_hit_modules.iter().cloned());
        union.extend(core_tactic_pack.iter().cloned());
        self.reduce_import_dag(&union)
    }
}

/// Scalar Quantizer (SQ8) parameters per dimension.
#[derive(Debug, Clone)]
pub struct Sq8Quantizer {
    pub mins: Vec<f32>,
    pub maxs: Vec<f32>,
}

impl Sq8Quantizer {
    pub fn quantize(&self, _vec: &[f32]) -> Result<Vec<u8>, VdbError> {
        todo!("TASK-M1-004: implement SQ8 quantization")
    }

    pub fn dequantize(&self, _codes: &[u8]) -> Result<Vec<f32>, VdbError> {
        todo!("TASK-M1-004: implement SQ8 dequantization")
    }

    pub fn asymmetric_l2(&self, _query: &[f32], _codes: &[u8]) -> Result<f32, VdbError> {
        todo!("TASK-M1-004: implement SQ8 asymmetric L2 distance")
    }
}

/// In-memory / mmap static HNSW + SQ8 index.
#[derive(Debug, Clone)]
pub struct StaticVdbIndex {
    pub dim: usize,
    pub quantizer: Sq8Quantizer,
    pub metadata: Vec<LemmaMetadata>,
    pub codes: Vec<Vec<u8>>,
}

impl StaticVdbIndex {
    /// Query top-k lemmas matching `target_head_symbol` or `"POLYMORPHIC"`.
    pub fn search_top_k(
        &self,
        _query: &[f32],
        _target_head_symbol: &str,
        _top_k: usize,
    ) -> Result<Vec<VdbSearchHit>, VdbError> {
        todo!("TASK-M1-006: implement filtered HNSW+SQ8 top-k search")
    }
}

/// Outcome of checking the per-theorem episodic subgoal cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodicHit {
    Novel,
    PruneLoop,
    Solved(Vec<String>),
}

/// Per-theorem episodic cache for loop pruning (`cos > 0.98`) and proof reuse.
#[derive(Debug, Default)]
pub struct EpisodicCache {
    pub entries: Vec<(Vec<f32>, Option<Vec<String>>)>,
}

impl EpisodicCache {
    pub fn check_or_insert(&mut self, _state_emb: &[f32]) -> EpisodicHit {
        todo!("TASK-M1-007: implement episodic cache >0.98 cosine loop pruning")
    }

    pub fn mark_solved(&mut self, _state_emb: Vec<f32>, _proof_script: Vec<String>) {
        todo!("TASK-M1-007: implement episodic cache mark_solved")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_episodic_threshold_constant() {
        assert!((EPISODIC_LOOP_THRESHOLD - 0.98).abs() < f32::EPSILON);
    }

    #[test]
    fn test_syntactic_closure_symbol_to_module() {
        let mut resolver = ImportResolver::default();
        resolver
            .symbol_to_module
            .insert("Finset".into(), "Mathlib.Data.Finset.Basic".into());
        resolver
            .symbol_to_module
            .insert("Nat.Prime".into(), "Mathlib.Data.Nat.Prime.Basic".into());
        resolver
            .symbol_to_module
            .insert("Nat".into(), "Init.Prelude".into());

        let mods = resolver.resolve_syntactic_modules("∀ (s : Finset Nat) (p : Nat), Nat.Prime p → p ∈ s");
        assert!(mods.contains("Mathlib.Data.Finset.Basic"));
        assert!(mods.contains("Mathlib.Data.Nat.Prime.Basic"));
        assert!(mods.contains("Init.Prelude"));
    }

    #[test]
    fn test_import_dag_transitive_reduction() {
        let mut resolver = ImportResolver::default();
        resolver
            .import_dag
            .insert("Mathlib.Tactic.Ring".into(), vec!["Mathlib.Data.Nat.Prime.Basic".into()]);
        resolver
            .import_dag
            .insert("Mathlib.Data.Nat.Prime.Basic".into(), vec!["Mathlib.Data.Nat.Basic".into()]);
        resolver
            .import_dag
            .insert("Mathlib.Data.Nat.Basic".into(), vec!["Init".into()]);

        let candidates: HashSet<String> = [
            "Init",
            "Mathlib.Data.Nat.Basic",
            "Mathlib.Data.Nat.Prime.Basic",
            "Mathlib.Tactic.Ring",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let reduced = resolver.reduce_import_dag(&candidates);
        assert_eq!(reduced, vec!["Mathlib.Tactic.Ring".to_string()]);
    }

    #[test]
    fn test_preflight_dynamic_module_selection() {
        let mut resolver = ImportResolver::default();
        resolver
            .symbol_to_module
            .insert("Finset".into(), "Mathlib.Data.Finset.Basic".into());
        resolver.import_dag.insert(
            "Mathlib.Data.Finset.Basic".into(),
            vec!["Algebra.Group.Defs".into(), "Init".into()],
        );
        let selected = resolver.resolve_preflight_modules(
            "Finset",
            &["Algebra.Group.Defs".into()],
            &["Init".into(), "Mathlib.Tactic.Ring".into()],
        );
        assert_eq!(
            selected,
            vec![
                "Mathlib.Data.Finset.Basic".to_string(),
                "Mathlib.Tactic.Ring".to_string()
            ]
        );
    }

    #[test]
    #[ignore = "TDD: TASK-M1-004"]
    fn test_sq8_quantization_roundtrip_and_asymmetric_l2() {
        let q = Sq8Quantizer {
            mins: vec![-1.0; 384],
            maxs: vec![1.0; 384],
        };
        let v: Vec<f32> = (0..384).map(|i| ((i as f32) / 384.0) - 0.5).collect();
        let codes = q.quantize(&v).unwrap();
        assert_eq!(codes.len(), 384);
        let recon = q.dequantize(&codes).unwrap();
        let l2 = q.asymmetric_l2(&v, &codes).unwrap();
        assert!(l2 < 1e-2);
        assert_eq!(recon.len(), 384);
    }

    #[test]
    #[ignore = "TDD: TASK-M1-006"]
    fn test_hnsw_filtered_head_symbol_search() {
        let idx = StaticVdbIndex {
            dim: 4,
            quantizer: Sq8Quantizer {
                mins: vec![-1.0; 4],
                maxs: vec![1.0; 4],
            },
            metadata: vec![
                LemmaMetadata {
                    name: "Nat.le_refl".into(),
                    head_symbol: "LE.le".into(),
                    module: "Init".into(),
                    arity: 1,
                    signature: "n ≤ n".into(),
                },
                LemmaMetadata {
                    name: "Nat.add_comm".into(),
                    head_symbol: "Eq".into(),
                    module: "Init".into(),
                    arity: 2,
                    signature: "a + b = b + a".into(),
                },
            ],
            codes: vec![vec![200, 200, 200, 200], vec![180, 180, 180, 180]],
        };
        let hits = idx.search_top_k(&[0.5, 0.5, 0.5, 0.5], "Eq", 16).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].metadata.name, "Nat.add_comm");
    }

    #[test]
    #[ignore = "TDD: TASK-M1-007"]
    fn test_episodic_subgoal_cache_loop_pruning() {
        let mut cache = EpisodicCache::default();
        let s1 = vec![1.0, 0.0, 0.0];
        let s2 = vec![0.999, 0.04, 0.0]; // cos > 0.98
        let s3 = vec![0.0, 1.0, 0.0]; // orthogonal

        assert_eq!(cache.check_or_insert(&s1), EpisodicHit::Novel);
        assert_eq!(cache.check_or_insert(&s2), EpisodicHit::PruneLoop);
        assert_eq!(cache.check_or_insert(&s3), EpisodicHit::Novel);
    }
}
