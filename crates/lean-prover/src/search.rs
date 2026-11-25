// SPDX-License-Identifier: AGPL-3.0-only
//! Deep (>= 3-step) Best-First compositional search tree with two-tier deduplication.

use crate::vfs::{BlobHash, BlobStore, TreeSnapshot};
use lean_sys1_ort::DifficultyBucket;
use lean_vdb::{EpisodicCache, EpisodicHit};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::path::PathBuf;

/// Arena-allocated Best-First search node holding an immutable `TreeSnapshot`.
#[derive(Clone, Debug)]
pub struct SearchNode {
    pub id: usize,
    pub parent: Option<usize>,
    pub depth: usize,
    pub tree: TreeSnapshot,
    pub target_file: PathBuf,
    pub open_subgoal: String,
    pub parent_subgoal: Option<String>,
    pub partial_script: Vec<String>,
    pub priority: f32,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.priority.to_bits() == other.priority.to_bits()
    }
}

impl Eq for SearchNode {}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// Priority formula: `w_diff(DifficultyBucket) + answer_confidence - 0.15 * depth`.
#[must_use]
pub fn compute_node_priority(
    difficulty: DifficultyBucket,
    answer_confidence: f32,
    depth: usize,
) -> f32 {
    let w_diff = match difficulty {
        DifficultyBucket::Trivial => 2.0,
        DifficultyBucket::Short => 1.0,
        DifficultyBucket::NeedsLemma => 0.0,
        DifficultyBucket::DeadEnd => -2.0,
    };
    w_diff + answer_confidence - 0.15 * (depth as f32)
}

/// Two-tier deduplication tracker: Tier 1 exact `(tree_hash, canon_subgoal)` before `Lean-Sys1`,
/// Tier 2 semantic `EpisodicCache` (`cos > 0.98`) after `Lean-Sys1`.
#[derive(Default, Debug)]
pub struct TwoTierDedup {
    pub visited_exact: HashSet<(BlobHash, String)>,
    pub visited_subgoals: HashSet<String>,
    pub episodic: EpisodicCache,
    pub exact_dedup_hits: usize,
    pub semantic_dedup_hits: usize,
    pub sys1_calls: usize,
}

impl TwoTierDedup {
    /// Tier-1 exact check before invoking `Lean-Sys1`. Returns `true` if novel, `false` if duplicate.
    pub fn check_tier1_exact(&mut self, tree_hash: BlobHash, canon_subgoal: &str) -> bool {
        if self
            .visited_exact
            .insert((tree_hash, canon_subgoal.to_string()))
        {
            true
        } else {
            self.exact_dedup_hits += 1;
            false
        }
    }

    /// Record a `Lean-Sys1` evaluation and run Tier-2 semantic dedup (`cos > 0.98`).
    pub fn record_sys1_and_check_tier2(&mut self, state_emb: &[f32]) -> bool {
        self.sys1_calls += 1;
        match self.episodic.check_or_insert(state_emb) {
            EpisodicHit::Novel => true,
            EpisodicHit::PruneLoop | EpisodicHit::Solved(_) => {
                self.semantic_dedup_hits += 1;
                false
            }
        }
    }

    /// Record a `Lean-Sys1` evaluation while never semantically pruning a child whose
    /// canonical goal string strictly changed or shrank relative to its parent.
    pub fn record_sys1_and_check_tier2_with_progress(
        &mut self,
        state_emb: &[f32],
        parent_subgoal: Option<&str>,
        open_subgoal: &str,
    ) -> bool {
        self.sys1_calls += 1;
        if let Some(parent) = parent_subgoal
            && parent == open_subgoal
        {
            self.semantic_dedup_hits += 1;
            return false;
        }
        let is_fresh_progress = parent_subgoal.is_some_and(|p| p != open_subgoal)
            && self.visited_subgoals.insert(open_subgoal.to_string());
        match self.episodic.check_or_insert(state_emb) {
            EpisodicHit::Novel => true,
            EpisodicHit::PruneLoop | EpisodicHit::Solved(_) => {
                if is_fresh_progress {
                    true
                } else {
                    self.semantic_dedup_hits += 1;
                    false
                }
            }
        }
    }
}

/// Outcome of a deep Best-First compositional search over `TreeSnapshot`s.
#[derive(Debug, Clone)]
pub struct DeepSearchResult {
    pub solved_tree: Option<TreeSnapshot>,
    pub winning_script: Option<Vec<String>>,
    pub expanded_nodes: usize,
    pub exact_dedup_hits: usize,
    pub sys1_calls: usize,
    pub last_diag: String,
}

/// Run bounded Best-First search (`max_depth`, `max_nodes`) using a caller-supplied step evaluator.
pub fn run_best_first_tree_search<F>(
    store: &mut BlobStore,
    root_tree: TreeSnapshot,
    target_file: PathBuf,
    initial_subgoal: String,
    max_depth: usize,
    max_nodes: usize,
    mut expand_step: F,
) -> DeepSearchResult
where
    F: FnMut(
        &mut BlobStore,
        &SearchNode,
        &mut TwoTierDedup,
    ) -> Result<StepOutcome, String>,
{
    let mut frontier = BinaryHeap::new();
    let mut dedup = TwoTierDedup::default();
    let mut next_id = 1usize;
    let mut expanded_nodes = 0usize;
    let mut last_diag = String::from("search frontier exhausted");

    frontier.push(SearchNode {
        id: 0,
        parent: None,
        depth: 0,
        tree: root_tree,
        target_file,
        open_subgoal: initial_subgoal,
        parent_subgoal: None,
        partial_script: Vec::new(),
        priority: 2.0,
    });

    while let Some(node) = frontier.pop() {
        if node.depth >= max_depth || expanded_nodes >= max_nodes {
            continue;
        }
        if !dedup.check_tier1_exact(node.tree.tree_hash, &node.open_subgoal) {
            continue;
        }
        expanded_nodes += 1;

        match expand_step(store, &node, &mut dedup) {
            Ok(StepOutcome::Solved { tree, script }) => {
                return DeepSearchResult {
                    solved_tree: Some(tree),
                    winning_script: Some(script),
                    expanded_nodes,
                    exact_dedup_hits: dedup.exact_dedup_hits,
                    sys1_calls: dedup.sys1_calls,
                    last_diag: String::new(),
                };
            }
            Ok(StepOutcome::Progressed(children)) => {
                for mut child in children {
                    if child.depth <= max_depth {
                        child.id = next_id;
                        child.parent = Some(node.id);
                        next_id += 1;
                        frontier.push(child);
                    }
                }
            }
            Ok(StepOutcome::Pruned(reason)) | Err(reason) => {
                if !reason.is_empty() {
                    last_diag = reason;
                }
            }
        }
    }

    DeepSearchResult {
        solved_tree: None,
        winning_script: None,
        expanded_nodes,
        exact_dedup_hits: dedup.exact_dedup_hits,
        sys1_calls: dedup.sys1_calls,
        last_diag,
    }
}

/// Step expansion result returned by the hybrid kernel evaluator.
#[derive(Debug, Clone)]
pub enum StepOutcome {
    Solved {
        tree: TreeSnapshot,
        script: Vec<String>,
    },
    Progressed(Vec<SearchNode>),
    Pruned(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m27_parent_child_progress_bypasses_semantic_prune() {
        let mut dedup = TwoTierDedup::default();
        let parent_emb = vec![1.0_f32; 8];
        let child_emb = vec![0.999_f32; 8];
        assert!(dedup.record_sys1_and_check_tier2_with_progress(
            &parent_emb,
            None,
            "⊢ U * Q = V"
        ));
        assert!(dedup.record_sys1_and_check_tier2_with_progress(
            &child_emb,
            Some("⊢ U * Q = V"),
            "⊢ ∀ i j, (U * Q) i j = V i j"
        ));
        assert!(!dedup.record_sys1_and_check_tier2_with_progress(
            &child_emb,
            Some("⊢ U * Q = V"),
            "⊢ U * Q = V"
        ));
    }
}
