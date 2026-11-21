use lean_prover::{TheoremProblem, prove_and_verify_with_lean4};
use lean_repl::{ReplError, verify_theorem_with_lean4_kernel};
use lean_sys1_ort::{CalibrationTemperatures, PremiseFitWeights};
use lean_vdb::{LemmaMetadata, Sq8Quantizer, StaticVdbIndex};
use orion::database::{Database, VectorDatabase};

fn build_lean4_benchmark_vdb() -> (StaticVdbIndex, VectorDatabase) {
    let dim = 8;
    let metadata = vec![
        LemmaMetadata {
            name: "Nat.add_assoc".into(),
            head_symbol: "Eq".into(),
            module: "Init.Data.Nat.Basic".into(),
            arity: 3,
            signature: "(a + b) + c = a + (b + c)".into(),
        },
        LemmaMetadata {
            name: "Nat.add_comm".into(),
            head_symbol: "Eq".into(),
            module: "Init.Data.Nat.Basic".into(),
            arity: 2,
            signature: "a + b = b + a".into(),
        },
        LemmaMetadata {
            name: "Nat.le_trans".into(),
            head_symbol: "LE.le".into(),
            module: "Init.Data.Nat.Basic".into(),
            arity: 2,
            signature: "a ≤ b → b ≤ c → a ≤ c".into(),
        },
        LemmaMetadata {
            name: "Nat.le_add_right".into(),
            head_symbol: "LE.le".into(),
            module: "Init.Data.Nat.Basic".into(),
            arity: 2,
            signature: "n ≤ n + k".into(),
        },
    ];

    let raw_vecs = [
        vec![0.9_f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.8_f32, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0_f32, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        vec![0.0_f32, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
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
        "lean_prover_test_orion_bench_{unique}_{:?}.vdb",
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
        },
        db,
    )
}

#[test]
fn test_e2e_lean4_benchmark_problem_1_nat_add_right_comm() {
    let (vdb, orion_db) = build_lean4_benchmark_vdb();
    let dim = 8;
    let w1 = vec![0.25_f32; dim * 4 * dim];
    let b1 = vec![0.0_f32; dim];
    let w2 = vec![0.25_f32; dim];
    let weights = PremiseFitWeights {
        w1: &w1,
        b1: &b1,
        w2: &w2,
        b2: 0.0,
    };
    let temps = CalibrationTemperatures {
        t_choice: 1.0,
        t_noul: 1.0,
        t_fit: 1.0,
    };

    let problem = TheoremProblem {
        name: "bench_nat_add_right_comm".into(),
        declaration: "theorem bench_nat_add_right_comm (a b c : Nat) : (a + b) + c = (a + c) + b".into(),
        hypotheses: vec!["a : Nat".into(), "b : Nat".into(), "c : Nat".into()],
        target: "(a + b) + c = (a + c) + b".into(),
    };
    let state_emb = vec![0.85_f32, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let report = prove_and_verify_with_lean4(
        &problem,
        &vdb,
        &orion_db,
        weights,
        &temps,
        &state_emb,
        &[0.1, 2.2, 0.2, -0.5, -0.5],
    )
    .expect("Problem 1 should be proved and verified by Lean 4");

    assert!(report.kernel_verified);
    assert_eq!(report.head_symbol, "Eq");
    println!("[BENCH 1] {report:?}");
}

#[test]
fn test_e2e_lean4_benchmark_problem_2_nat_le_trans_add() {
    let (vdb, orion_db) = build_lean4_benchmark_vdb();
    let dim = 8;
    let w1 = vec![0.25_f32; dim * 4 * dim];
    let b1 = vec![0.0_f32; dim];
    let w2 = vec![0.25_f32; dim];
    let weights = PremiseFitWeights {
        w1: &w1,
        b1: &b1,
        w2: &w2,
        b2: 0.0,
    };
    let temps = CalibrationTemperatures {
        t_choice: 1.0,
        t_noul: 1.0,
        t_fit: 1.0,
    };

    let problem = TheoremProblem {
        name: "bench_nat_le_trans_add".into(),
        declaration: "theorem bench_nat_le_trans_add (a b c d e : Nat) (h_unused : e = e) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c + d".into(),
        hypotheses: vec![
            "a : Nat".into(),
            "b : Nat".into(),
            "c : Nat".into(),
            "d : Nat".into(),
            "e : Nat".into(),
            "h_unused : e = e".into(),
            "h1 : a ≤ b".into(),
            "h2 : b ≤ c".into(),
        ],
        target: "a ≤ c + d".into(),
    };
    let state_emb = vec![0.0_f32, 0.0, 0.85, 0.15, 0.0, 0.0, 0.0, 0.0];
    let report = prove_and_verify_with_lean4(
        &problem,
        &vdb,
        &orion_db,
        weights,
        &temps,
        &state_emb,
        &[0.2, 0.15, 0.1, -0.2, -0.2], // <0.50 fallback gate triggered
    )
    .expect("Problem 2 should be proved and verified by Lean 4");

    assert!(report.kernel_verified);
    assert_eq!(report.head_symbol, "LE.le");
    assert!(!report.pruned_hypotheses.iter().any(|h| h.contains("h_unused")));
    println!("[BENCH 2] {report:?}");
}

#[test]
fn test_e2e_lean4_kernel_rejects_sorry_and_invalid_proof() {
    let sorry_res = verify_theorem_with_lean4_kernel(
        "bad_sorry",
        "theorem bad_sorry (n : Nat) : n = n + 1",
        &["sorry".to_string()],
    );
    assert!(matches!(sorry_res, Err(ReplError::SorryAxiomDetected)));

    let invalid_res = verify_theorem_with_lean4_kernel(
        "bad_rfl",
        "theorem bad_rfl (n : Nat) : n = n + 1",
        &["rfl".to_string()],
    );
    assert!(matches!(invalid_res, Err(ReplError::SorryAxiomDetected)) || matches!(invalid_res, Ok(r) if !r.success));
}

#[test]
fn test_e2e_eval_workspace_all_three_problems_and_anticheat() {
    let ws = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("eval_workspace");
    let entries = lean_prover::load_eval_manifest(&ws.join("manifest.jsonl")).unwrap();
    let (vdb, orion_db) = build_lean4_benchmark_vdb();
    let dim = 8;
    let w1 = vec![0.25_f32; dim * 4 * dim];
    let b1 = vec![0.0_f32; dim];
    let w2 = vec![0.25_f32; dim];
    let weights = PremiseFitWeights {
        w1: &w1,
        b1: &b1,
        w2: &w2,
        b2: 0.0,
    };
    let temps = CalibrationTemperatures {
        t_choice: 1.0,
        t_noul: 1.0,
        t_fit: 1.0,
    };
    let state_emb = vec![0.85_f32, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    for entry in &entries {
        let report = lean_prover::solve_eval_workspace_entry(
            &ws,
            entry,
            &vdb,
            &orion_db,
            weights,
            &temps,
            &state_emb,
            &[0.1, 2.2, 0.2, -0.5, -0.5],
        )
        .unwrap_or_else(|e| panic!("failed {}: {e}", entry.id));
        assert!(report.kernel_verified);
        println!("[EVAL {}] {:?}", entry.id, report);
    }
}
