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
    #[error("orion backend error: {0}")]
    Backend(#[from] orion::error::VdbError),
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
    pub fn quantize(&self, vec: &[f32]) -> Result<Vec<u8>, VdbError> {
        if vec.len() != self.mins.len() {
            return Err(VdbError::DimensionMismatch {
                expected: self.mins.len(),
                got: vec.len(),
            });
        }
        Ok(vec
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let span = (self.maxs[i] - self.mins[i]).max(1e-12);
                let norm = ((x - self.mins[i]) / span).clamp(0.0, 1.0);
                (norm * 255.0).round() as u8
            })
            .collect())
    }

    pub fn dequantize(&self, codes: &[u8]) -> Result<Vec<f32>, VdbError> {
        if codes.len() != self.mins.len() {
            return Err(VdbError::DimensionMismatch {
                expected: self.mins.len(),
                got: codes.len(),
            });
        }
        Ok(codes
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let span = self.maxs[i] - self.mins[i];
                self.mins[i] + (f32::from(c) / 255.0) * span
            })
            .collect())
    }

    pub fn asymmetric_l2(&self, query: &[f32], codes: &[u8]) -> Result<f32, VdbError> {
        if query.len() != self.mins.len() || codes.len() != self.mins.len() {
            return Err(VdbError::DimensionMismatch {
                expected: self.mins.len(),
                got: query.len().min(codes.len()),
            });
        }
        let sum_sq: f32 = query
            .iter()
            .zip(codes.iter())
            .enumerate()
            .map(|(i, (&q, &c))| {
                let span = self.maxs[i] - self.mins[i];
                let recon = self.mins[i] + (f32::from(c) / 255.0) * span;
                let diff = q - recon;
                diff * diff
            })
            .sum();
        Ok(sum_sq.sqrt())
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
        query: &[f32],
        target_head_symbol: &str,
        top_k: usize,
    ) -> Result<Vec<VdbSearchHit>, VdbError> {
        if query.len() != self.dim {
            return Err(VdbError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut hits = Vec::new();
        for (meta, code) in self.metadata.iter().zip(self.codes.iter()) {
            if meta.head_symbol != target_head_symbol && meta.head_symbol != "POLYMORPHIC" {
                continue;
            }
            let distance = self.quantizer.asymmetric_l2(query, code)?;
            let embedding = self.quantizer.dequantize(code)?;
            hits.push(VdbSearchHit {
                metadata: meta.clone(),
                distance,
                embedding,
            });
        }
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Query an `orion` `VectorDatabase` using `search_with_filter` for `target_head_symbol` or `"POLYMORPHIC"`.
    pub fn search_hnsw_backend(
        &self,
        db: &orion::database::VectorDatabase,
        query: &[f32],
        target_head_symbol: &str,
        top_k: usize,
    ) -> Result<Vec<VdbSearchHit>, VdbError> {
        let raw_hits = db.search_with_filter(query, top_k, |id| {
            self.metadata.get(id as usize).is_some_and(|meta| {
                meta.head_symbol == target_head_symbol || meta.head_symbol == "POLYMORPHIC"
            })
        })?;

        let mut hits = Vec::with_capacity(raw_hits.len());
        for (id, distance) in raw_hits {
            let idx = id as usize;
            if let (Some(meta), Some(code)) = (self.metadata.get(idx), self.codes.get(idx)) {
                hits.push(VdbSearchHit {
                    metadata: meta.clone(),
                    distance,
                    embedding: self.quantizer.dequantize(code)?,
                });
            }
        }
        Ok(hits)
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
    pub fn check_or_insert(&mut self, state_emb: &[f32]) -> EpisodicHit {
        for (visited_emb, solved_script) in &self.entries {
            if cosine_sim(state_emb, visited_emb) > EPISODIC_LOOP_THRESHOLD {
                return match solved_script {
                    Some(script) => EpisodicHit::Solved(script.clone()),
                    None => EpisodicHit::PruneLoop,
                };
            }
        }
        self.entries.push((state_emb.to_vec(), None));
        EpisodicHit::Novel
    }

    pub fn mark_solved(&mut self, state_emb: Vec<f32>, proof_script: Vec<String>) {
        self.entries.push((state_emb, Some(proof_script)));
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    orion::types::cosine_similarity(a, b).unwrap_or(0.0)
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
        assert!(l2 < 0.1);
        assert_eq!(recon.len(), 384);
    }

    #[test]
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
