// SPDX-License-Identifier: AGPL-3.0-only
//! Export theorem signatures from a Lean 4 environment (`Init` or `Mathlib`) and pack `mathlib_sq8.vdb`.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const LEAN_EXPORTER_TEMPLATE: &str = r#"
{IMPORTS}
open Lean Meta in
#eval show MetaM Unit from do
  let env ← getEnv
  let mut count := 0
  let maxCount := {MAX_LEMMAS}
  for (name, cinfo) in env.constants.toList do
    if count >= maxCount then break
    let .thmInfo val := cinfo | continue
    let s := name.toString
    if name.isInternal || s.startsWith "_" || (s.splitOn "._").length > 1 || (s.splitOn ".match_").length > 1 || (s.splitOn ".proof_").length > 1 then
      continue
    let modIdx? := env.getModuleIdxFor? name
    let modName := match modIdx? with
      | some idx => (env.header.moduleNames[idx]!).toString
      | none => "{DEFAULT_MOD}"
    let tyStr := (← ppExpr val.type).pretty 200
    if tyStr.length > 240 then continue
    let jsonLine := Json.mkObj [
      ("name", Json.str s),
      ("module", Json.str modName),
      ("signature", Json.str (tyStr.replace "\n" " "))
    ]
    IO.println s!"__VDB_LEMMA__{jsonLine.compress}"
    count := count + 1
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut lake_dir: Option<PathBuf> = None;
    let mut max_lemmas: usize = 1500;
    let mut out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("mathlib_sq8.vdb");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lake-dir" if i + 1 < args.len() => {
                lake_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--max-lemmas" if i + 1 < args.len() => {
                max_lemmas = args[i + 1].parse()?;
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let (imports, default_mod, mut cmd) = if let Some(dir) = &lake_dir {
        let mut c = Command::new("lake");
        c.args(["env", "lean", "--stdin"]).current_dir(dir);
        ("import Mathlib\nimport Lean", "Mathlib", c)
    } else {
        let mut c = Command::new("lean");
        c.arg("--stdin");
        ("import Init\nimport Lean", "Init", c)
    };

    let script = LEAN_EXPORTER_TEMPLATE
        .replace("{IMPORTS}", imports)
        .replace("{MAX_LEMMAS}", &max_lemmas.to_string())
        .replace("{DEFAULT_MOD}", default_mod);

    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(script.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    let emb = lean_prover::embedded_artifacts();
    let onnx_sess = lean_sys1_ort::LeanSys1OnnxSession::from_onnx_bytes(emb.lean_sys1_onnx)?;
    let temps: lean_sys1_ort::CalibrationTemperatures =
        serde_json::from_str(emb.temperatures_json)?;

    let mut seen = HashSet::new();
    let mut entries: Vec<(lean_vdb::LemmaMetadata, Vec<f32>)> = Vec::new();
    let mut resolver = lean_vdb::ImportResolver::default();

    for line in stdout.lines() {
        let Some(payload) = line.strip_prefix("__VDB_LEMMA__") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        let name = v.get("name").and_then(|s| s.as_str()).unwrap_or("").trim();
        let module = v
            .get("module")
            .and_then(|s| s.as_str())
            .unwrap_or(default_mod)
            .trim();
        let sig = v
            .get("signature")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .trim();
        if name.is_empty() || sig.is_empty() || !seen.insert(name.to_string()) {
            continue;
        }
        let Ok(goal) = lean_canonicalizer::canonicalize_goal(&[], sig) else {
            continue;
        };
        let text = format!("{} ⊢ {}", goal.hypotheses.join(", "), goal.target);
        let Ok((ids, mask)) = lean_sys1_ort::encode_canonical_text(emb.tokenizer_json, &text, 512)
        else {
            continue;
        };
        let Ok((pred, _, _)) = onnx_sess.forward_with_gate(&ids, &mask, &temps, 0.5) else {
            continue;
        };
        resolver
            .symbol_to_module
            .insert(name.to_string(), module.to_string());
        entries.push((
            lean_vdb::LemmaMetadata {
                name: name.to_string(),
                head_symbol: goal.head_symbol.as_str().to_string(),
                module: module.to_string(),
                arity: 1,
                signature: sig.to_string(),
            },
            pred.state_emb,
        ));
    }

    if entries.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Lean exporter produced 0 lemmas.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
        .into());
    }

    let dim = 384usize;
    let mut mins = vec![f32::INFINITY; dim];
    let mut maxs = vec![f32::NEG_INFINITY; dim];
    for (_, e) in &entries {
        for (j, &val) in e.iter().enumerate() {
            if val < mins[j] {
                mins[j] = val;
            }
            if val > maxs[j] {
                maxs[j] = val;
            }
        }
    }
    let quantizer = lean_vdb::Sq8Quantizer { mins, maxs };
    let mut idx = lean_vdb::StaticVdbIndex {
        dim,
        quantizer,
        metadata: Vec::with_capacity(entries.len()),
        codes: Vec::with_capacity(entries.len()),
        neighbors: Vec::new(),
    };
    idx.extend_with_dynamic_entries(entries)?;
    let bytes = idx.to_lvdb_bytes(&resolver);
    std::fs::write(&out_path, &bytes)?;
    println!(
        "Packed {} lemmas into {} ({} bytes)",
        idx.metadata.len(),
        out_path.display(),
        bytes.len()
    );
    Ok(())
}
