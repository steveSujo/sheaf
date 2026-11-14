use lean_canonicalizer::{
    HeadSymbol, collapse_implicit_instances, extract_head_symbol, normalize_alpha_vars,
    prune_disjoint_hypotheses,
};
use lean_prover::{
    CORE_TACTIC_KEYWORDS, RepairDecision, build_pda_logit_mask, evaluate_error_repair,
    plan_prefix_replay_modules, project_soft_tokens,
};
use lean_repl::{Goal, GoalGroup, PantographSession, partition_coupled_goals, verify_zero_sorry};
use lean_sys1_ort::{
    CalibrationTemperatures, FallbackOutcome, GateDecision, PremiseFitWeights, StrategyChoice,
    encode_canonical_text, route_with_confidence_gate, score_premise_fit_batch,
    select_fallback_outcome,
};
use lean_vdb::{EpisodicCache, EpisodicHit, ImportResolver, LemmaMetadata, Sq8Quantizer, StaticVdbIndex};
use orion::database::{Database, VectorDatabase};
use std::collections::HashMap;
use std::io::Cursor;

fn make_test_vdb() -> (StaticVdbIndex, VectorDatabase) {
    let dim = 8;
    let metadata = vec![
        LemmaMetadata {
            name: "Nat.add_comm".into(),
            head_symbol: "Eq".into(),
            module: "Init.Data.Nat.Basic".into(),
            arity: 2,
            signature: "a + b = b + a".into(),
        },
        LemmaMetadata {
            name: "Nat.add_assoc".into(),
            head_symbol: "Eq".into(),
            module: "Init.Data.Nat.Basic".into(),
            arity: 3,
            signature: "(a + b) + c = a + (b + c)".into(),
        },
        LemmaMetadata {
            name: "Nat.le_trans".into(),
            head_symbol: "LE.le".into(),
            module: "Init.Data.Nat.Basic".into(),
            arity: 2,
            signature: "a ≤ b → b ≤ c → a ≤ c".into(),
        },
        LemmaMetadata {
            name: "id".into(),
            head_symbol: "POLYMORPHIC".into(),
            module: "Init.Core".into(),
            arity: 1,
            signature: "a → a".into(),
        },
    ];

    let raw_vecs = [
        vec![0.9_f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.8_f32, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0_f32, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        vec![0.5_f32, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let quantizer = Sq8Quantizer {
        mins: vec![0.0; dim],
        maxs: vec![1.0; dim],
    };
    let codes: Vec<Vec<u8>> = raw_vecs
        .iter()
        .map(|v| quantizer.quantize(v).unwrap())
        .collect();

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let db_path = std::env::temp_dir().join(format!(
        "lean_prover_test_orion_integ_{unique}_{:?}.vdb",
        std::thread::current().id()
    ));
    let mut db = VectorDatabase::open(&db_path, None).unwrap();
    for (i, v) in raw_vecs.iter().enumerate() {
        db.insert(i as u64, orion::types::Vector { data: v.clone() }).unwrap();
    }

    (
        StaticVdbIndex {
            dim,
            quantizer,
            metadata,
            codes,
            neighbors: Vec::new(),
        },
        db,
    )
}

#[test]
fn test_integration_canonicalizer_to_sys1_ort_pipeline() {
    let hyps = vec![
        "a : Nat".to_string(),
        "b : Nat".to_string(),
        "unused : z = z".to_string(),
    ];
    let raw_target = "Nat.add inst✝¹ a b = Nat.add b a";
    let pruned = prune_disjoint_hypotheses(&hyps, raw_target, 6);
    assert_eq!(pruned.len(), 2);

    let collapsed = collapse_implicit_instances(raw_target);
    assert!(collapsed.contains("[_]"));

    let (norm_hyps, norm_target) = normalize_alpha_vars(&pruned, &collapsed);
    assert_eq!(extract_head_symbol(&norm_target).unwrap(), HeadSymbol::Named("Eq".into()));

    let canonical_text = format!("{} ⊢ {}", norm_hyps.join(", "), norm_target);
    let tok_json = r#"{
      "version": "1.0",
      "added_tokens": [
        {"id": 0, "content": "<pad>", "special": true},
        {"id": 1, "content": "<unk>", "special": true},
        {"id": 2, "content": "<s>", "special": true},
        {"id": 3, "content": "</s>", "special": true},
        {"id": 4, "content": "⊢", "special": true}
      ],
      "model": {
        "type": "BPE",
        "unk_token": "<unk>",
        "vocab": {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "⊢": 4, "_v0": 5, "_v1": 6},
        "merges": []
      }
    }"#;
    let (ids, mask) = encode_canonical_text(tok_json, &canonical_text, 32).unwrap();
    assert_eq!(ids.len(), 32);
    assert_eq!(mask.len(), 32);

    let temps: CalibrationTemperatures =
        serde_json::from_str(r#"{"T_choice": 1.0, "T_noul": 1.0, "T_fit": 1.0}"#).unwrap();
    let (strat, conf, dec) = route_with_confidence_gate(&[2.4, 0.2, 0.1, -0.5, -0.5], temps.t_choice);
    assert_eq!(strat, StrategyChoice::AtomicClose);
    assert!(conf >= 0.50);
    assert_eq!(dec, GateDecision::DirectExecute(StrategyChoice::AtomicClose));
}

#[test]
fn test_integration_canonicalizer_vdb_orion_to_premise_fit() {
    let (vdb, orion_db) = make_test_vdb();
    let head = extract_head_symbol("a + b = b + a").unwrap();
    assert_eq!(head, HeadSymbol::Named("Eq".into()));

    let query = vec![0.85_f32, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let hits = vdb
        .search_hnsw_backend(&orion_db, &query, head.as_str(), 4)
        .unwrap();
    assert!(hits.iter().any(|h| h.metadata.name == "Nat.add_comm"));
    assert!(hits.iter().all(|h| h.metadata.head_symbol == "Eq" || h.metadata.head_symbol == "POLYMORPHIC"));

    let cand_embs: Vec<Vec<f32>> = hits.iter().map(|h| h.embedding.clone()).collect();
    let d = query.len();
    let w1 = vec![0.25_f32; d * 4 * d];
    let b1 = vec![0.0_f32; d];
    let w2 = vec![0.25_f32; d];
    let weights = PremiseFitWeights {
        w1: &w1,
        b1: &b1,
        w2: &w2,
        b2: 0.0,
    };

    let scored = score_premise_fit_batch(&query, &cand_embs, weights, 1.0, 0.50).unwrap();
    match select_fallback_outcome(&scored, 2) {
        FallbackOutcome::BranchOnLemmas(top) => assert!(!top.is_empty() && top.len() <= 2),
        FallbackOutcome::ForceIntroduceHave => panic!("expected surviving lemmas"),
    }
}

#[test]
fn test_integration_vdb_import_resolver_prefix_replay_and_pda_mask() {
    let mut resolver = ImportResolver {
        symbol_to_module: HashMap::from([
            ("Nat".into(), "Init.Prelude".into()),
            ("Nat.add_comm".into(), "Init.Data.Nat.Basic".into()),
        ]),
        import_dag: HashMap::from([("Init.Data.Nat.Basic".into(), vec!["Init.Prelude".into()])]),
    };
    let _ = &mut resolver;

    let active = resolver.resolve_preflight_modules(
        "Nat.add_comm",
        &[],
        &["Init.Prelude".to_string()],
    );
    assert_eq!(active, vec!["Init.Data.Nat.Basic".to_string()]);

    let expanded = plan_prefix_replay_modules(&active, "Mathlib.Algebra.Ring.Basic").unwrap();
    assert!(expanded.contains(&"Mathlib.Algebra.Ring.Basic".to_string()));

    let soft = project_soft_tokens(&[vec![0.1_f32; 16], vec![0.2_f32; 16]], 3, 16);
    assert_eq!(soft.len(), 6);

    let vocab = vec![
        "Nat.add_comm".to_string(),
        "Nat.mul_comm".to_string(),
        CORE_TACTIC_KEYWORDS[0].to_string(),
    ];
    let mask = build_pda_logit_mask(&vocab, &[], &["Nat.add_comm".to_string()], "");
    assert_eq!(mask[0], 0.0);
    assert!(mask[1].is_infinite() && mask[1].is_sign_negative());
    assert_eq!(mask[2], 0.0);
}

#[test]
fn test_integration_repl_goal_coupling_repair_and_episodic_cache() {
    let g1 = Goal {
        goal_id: "g1".into(),
        target: "?m1 = 0".into(),
        head_symbol: "Eq".into(),
        hypotheses: vec![],
        metavars: vec!["?m1".into()],
    };
    let g2 = Goal {
        goal_id: "g2".into(),
        target: "?m1 ≤ 1".into(),
        head_symbol: "LE.le".into(),
        hypotheses: vec![],
        metavars: vec!["?m1".into()],
    };
    let g3 = Goal {
        goal_id: "g3".into(),
        target: "2 = 2".into(),
        head_symbol: "Eq".into(),
        hypotheses: vec![],
        metavars: vec![],
    };
    let groups = partition_coupled_goals(&[g1, g2, g3]);
    assert!(matches!(groups[0], GoalGroup::Sequential(_)));
    assert!(matches!(groups[1], GoalGroup::Independent(_)));

    let resp = r#"{"state_id": 1, "goals": []}"#.to_string() + "\n";
    let mut sess = PantographSession::new(Cursor::new(resp), Vec::new(), vec!["Init".into()]);
    let next_state = sess.execute_tactic(0, "g3", "rfl").unwrap();
    assert!(next_state.goals.is_empty());

    let mut cache = EpisodicCache::default();
    let emb = vec![1.0_f32, 0.0, 0.0, 0.0];
    assert_eq!(cache.check_or_insert(&emb), EpisodicHit::Novel);
    assert_eq!(cache.check_or_insert(&emb), EpisodicHit::PruneLoop);
    let mut solved_cache = EpisodicCache::default();
    solved_cache.mark_solved(emb.clone(), vec!["rfl".into()]);
    assert_eq!(solved_cache.check_or_insert(&emb), EpisodicHit::Solved(vec!["rfl".into()]));

    assert_eq!(
        evaluate_error_repair(0, "tactic failed"),
        RepairDecision::RetryWithDiagnostic("tactic failed".into())
    );
    assert_eq!(evaluate_error_repair(1, "tactic failed"), RepairDecision::MarkBranchFailed);
    assert!(verify_zero_sorry("theorem t depends on axioms: [propext]").unwrap());
}

#[test]
fn test_integration_eval_workspace_manifest_and_signature_lock() {
    let ws = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("eval_workspace");
    let entries = lean_prover::load_eval_manifest(&ws.join("manifest.jsonl")).unwrap();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].id, "P001_nat_add_comm");
    assert_eq!(entries[1].id, "P010_list_len_append");
    assert_eq!(entries[2].id, "P020_have_cutpoint");

    for entry in &entries {
        assert!(ws.join(&entry.file).exists(), "missing stub: {}", entry.file);
    }

    // Gate 1 Signature Lock must reject tampered theorem statements
    let sol_dir = ws.join("solutions");
    std::fs::create_dir_all(&sol_dir).unwrap();
    let tampered = "import Init\n\nnamespace Eval.P001\n\ntheorem nat_add_comm (a b : Nat) : a + b = a + b := by\n  rfl\n\nend Eval.P001\n";
    std::fs::write(sol_dir.join("P001_tampered.lean"), tampered).unwrap();
    let res = lean_repl::verify_solution_file_in_workspace(
        &ws,
        &entries[0].file,
        "solutions/P001_tampered.lean",
        &entries[0].theorem,
    );
    let _ = std::fs::remove_file(sol_dir.join("P001_tampered.lean"));
    assert!(res.is_err());
}
