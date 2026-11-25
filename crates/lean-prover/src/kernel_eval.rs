// SPDX-License-Identifier: AGPL-3.0-only
//! Hybrid Kernel Evaluator: Single-file batched probes with `maxHeartbeats` isolation
//! and Multi-file ephemeral dirty-worktree evaluation with early cancellation.

use crate::ui::{BranchStatus, UiEvent, UiHandle};
use crate::vfs::{BlobStore, TreeSnapshot};
use crate::{DiscoveredTheorem, ProverError, replace_theorem_sorry};
use std::cell::RefCell;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Result of batched candidate evaluation: `(solved_src_and_script, last_diag, unsolved_goal_diags)`.
pub type BatchedVerifyResult = (
    Option<(String, Vec<String>)>,
    String,
    Vec<(Vec<String>, String)>,
);

/// Abstraction over Lean 4 kernel verification backends (`BatchLeanCliBackend` or future persistent `PantographSession`).
pub trait KernelBackend {
    fn eval_batch(
        &self,
        src: &str,
        thm: &DiscoveredTheorem,
        full_name: &str,
        candidates: &[Vec<String>],
        lake_root: Option<&Path>,
        max_heartbeats: usize,
    ) -> Result<BatchedVerifyResult, ProverError>;

    fn audit_winner(
        &self,
        src: &str,
        thm_name: &str,
        full_name: &str,
        script: &[String],
        lake_root: Option<&Path>,
    ) -> Result<Result<String, String>, ProverError>;
}

/// Default `KernelBackend` using batched `lake env lean --json` probes with per-probe `maxHeartbeats`.
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchLeanCliBackend;

impl KernelBackend for BatchLeanCliBackend {
    fn eval_batch(
        &self,
        src: &str,
        thm: &DiscoveredTheorem,
        full_name: &str,
        candidates: &[Vec<String>],
        lake_root: Option<&Path>,
        max_heartbeats: usize,
    ) -> Result<BatchedVerifyResult, ProverError> {
        verify_candidates_batched_with_heartbeats(
            src,
            thm,
            full_name,
            candidates,
            lake_root,
            max_heartbeats,
        )
    }

    fn audit_winner(
        &self,
        src: &str,
        thm_name: &str,
        full_name: &str,
        script: &[String],
        lake_root: Option<&Path>,
    ) -> Result<Result<String, String>, ProverError> {
        audit_single_candidate(src, thm_name, full_name, script, lake_root)
    }
}

enum ServerEvalOutcome {
    Ndjson(String),
    TimedOut,
}

fn normalize_lake_root(dir: &Path) -> PathBuf {
    let abs = if dir.is_relative() {
        std::env::current_dir()
            .map(|cwd| cwd.join(dir))
            .unwrap_or_else(|_| dir.to_path_buf())
    } else {
        dir.to_path_buf()
    };
    let cleaned: PathBuf = abs.components().collect();
    let norm = std::fs::canonicalize(&cleaned).unwrap_or(cleaned);
    let raw = norm.to_string_lossy();
    if let Some(rest) = raw.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = raw.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        norm
    }
}

fn path_to_file_uri(dir: &Path, filename: &str) -> String {
    let joined = dir.join(filename);
    let raw = joined.to_string_lossy();
    let no_verbatim = raw
        .strip_prefix(r"\\?\UNC\")
        .map(|r| format!(r"\\{r}"))
        .or_else(|| raw.strip_prefix(r"\\?\").map(str::to_string))
        .unwrap_or_else(|| raw.into_owned());
    let slashes = no_verbatim.replace('\\', "/").replace(' ', "%20");
    if slashes.starts_with('/') {
        format!("file://{slashes}")
    } else {
        format!("file:///{slashes}")
    }
}

fn read_lsp_frame(reader: &mut BufReader<std::process::ChildStdout>) -> Option<serde_json::Value> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).ok()?;
        if n == 0 {
            return None;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':')
            && k.eq_ignore_ascii_case("Content-Length")
        {
            content_length = v.trim().parse::<usize>().ok();
        }
    }
    let len = content_length?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).ok()?;
    serde_json::from_slice::<serde_json::Value>(&buf).ok()
}

fn write_lsp_frame(stdin: &mut std::process::ChildStdin, msg: &serde_json::Value) -> Result<(), ProverError> {
    let body = serde_json::to_vec(msg).map_err(|e| {
        ProverError::Repl(lean_repl::ReplError::Io(std::io::Error::other(
            e.to_string(),
        )))
    })?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    stdin
        .write_all(header.as_bytes())
        .and_then(|_| stdin.write_all(&body))
        .and_then(|_| stdin.flush())
        .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))
}

struct LeanServerSession {
    lake_root: Option<PathBuf>,
    uri: String,
    version: u64,
    next_id: u64,
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    rx: mpsc::Receiver<serde_json::Value>,
}

impl Drop for LeanServerSession {
    fn drop(&mut self) {
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl LeanServerSession {
    fn spawn(lake_root: Option<&Path>, deadline: Option<Instant>) -> Result<Self, ProverError> {
        let norm_root = lake_root.map(normalize_lake_root);
        let mut cmd = if let Some(ref root) = norm_root {
            let mut c = std::process::Command::new("lake");
            c.args(["env", "lean", "--server"]);
            c.current_dir(root);
            c
        } else {
            let mut c = std::process::Command::new("lean");
            c.arg("--server");
            c
        };
        cmd.args([
            "-Dlinter.unusedVariables=true",
            "-Dlinter.unusedSimpArgs=true",
        ]);
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        let mut child = cmd
            .spawn()
            .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))?;
        let mut stdin = child.stdin.take().ok_or_else(|| {
            ProverError::Repl(lean_repl::ReplError::Io(std::io::Error::other(
                "missing lean --server stdin",
            )))
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            ProverError::Repl(lean_repl::ReplError::Io(std::io::Error::other(
                "missing lean --server stdout",
            )))
        })?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            while let Some(frame) = read_lsp_frame(&mut reader) {
                if tx.send(frame).is_err() {
                    break;
                }
            }
        });

        let init_msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": serde_json::Value::Null,
                "rootUri": serde_json::Value::Null,
                "capabilities": {}
            }
        });
        write_lsp_frame(&mut stdin, &init_msg)?;
        let default_init = Instant::now() + Duration::from_secs(30);
        let init_deadline = deadline.map_or(default_init, |dl| dl.min(default_init));
        loop {
            let now = Instant::now();
            if now >= init_deadline {
                return Err(ProverError::Repl(lean_repl::ReplError::Io(
                    std::io::Error::new(std::io::ErrorKind::TimedOut, "lean --server initialize timeout"),
                )));
            }
            let rem = init_deadline.saturating_duration_since(now);
            let frame = rx.recv_timeout(rem).map_err(|e| {
                ProverError::Repl(lean_repl::ReplError::Io(std::io::Error::other(format!(
                    "lean --server initialize timeout: {e}"
                ))))
            })?;
            if frame.get("id").and_then(|v| v.as_u64()) == Some(1) {
                break;
            }
        }
        write_lsp_frame(
            &mut stdin,
            &serde_json::json!({
                "jsonrpc": "2.0",
                "method": "initialized",
                "params": {}
            }),
        )?;

        let doc_dir = norm_root
            .clone()
            .unwrap_or_else(std::env::temp_dir);
        let uri = path_to_file_uri(&doc_dir, "__sheaf_live_worker.lean");
        Ok(Self {
            lake_root: norm_root,
            uri,
            version: 0,
            next_id: 2,
            child,
            stdin,
            rx,
        })
    }

    fn eval_document(
        &mut self,
        src: &str,
        deadline: Option<Instant>,
    ) -> Result<ServerEvalOutcome, ProverError> {
        self.version += 1;
        let ver = self.version;
        let msg = if ver == 1 {
            serde_json::json!({
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": self.uri,
                        "languageId": "lean4",
                        "version": ver,
                        "text": src
                    }
                }
            })
        } else {
            serde_json::json!({
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {
                        "uri": self.uri,
                        "version": ver
                    },
                    "contentChanges": [{ "text": src }]
                }
            })
        };
        write_lsp_frame(&mut self.stdin, &msg)?;

        let mut latest_diags: Vec<serde_json::Value> = Vec::new();
        let max_wait = deadline.unwrap_or_else(|| Instant::now() + Duration::from_secs(120));

        loop {
            let now = Instant::now();
            if now >= max_wait {
                return Ok(ServerEvalOutcome::TimedOut);
            }
            let wait_for = max_wait.saturating_duration_since(now);
            let frame = match self.rx.recv_timeout(wait_for) {
                Ok(f) => f,
                Err(mpsc::RecvTimeoutError::Timeout) => return Ok(ServerEvalOutcome::TimedOut),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(ProverError::Repl(lean_repl::ReplError::Io(
                        std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "lean --server stdout closed",
                        ),
                    )));
                }
            };

            if frame.get("method").is_some()
                && let Some(req_id) = frame.get("id")
            {
                let _ = write_lsp_frame(
                    &mut self.stdin,
                    &serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": serde_json::Value::Null
                    }),
                );
                continue;
            }

            let method = frame.get("method").and_then(|m| m.as_str()).unwrap_or("");
            if method == "textDocument/publishDiagnostics" {
                let msg_ver = frame
                    .get("params")
                    .and_then(|p| p.get("version"))
                    .and_then(|v| v.as_u64());
                if msg_ver == Some(ver) {
                    latest_diags = frame
                        .get("params")
                        .and_then(|p| p.get("diagnostics"))
                        .and_then(|d| d.as_array())
                        .cloned()
                        .unwrap_or_default();
                }
            } else if method == "$/lean/fileProgress" {
                let msg_ver = frame
                    .get("params")
                    .and_then(|p| p.get("textDocument"))
                    .and_then(|t| t.get("version"))
                    .and_then(|v| v.as_u64());
                let processing_empty = frame
                    .get("params")
                    .and_then(|p| p.get("processing"))
                    .and_then(|a| a.as_array())
                    .is_some_and(Vec::is_empty);
                if msg_ver == Some(ver) && processing_empty {
                    let drain_cap = if latest_diags.is_empty() { 500 } else { 30 };
                    let mut drain_until = Instant::now() + Duration::from_millis(drain_cap);
                    while Instant::now() < drain_until {
                        let rem = drain_until
                            .saturating_duration_since(Instant::now())
                            .min(max_wait.saturating_duration_since(Instant::now()));
                        if rem.is_zero() {
                            break;
                        }
                        let Ok(extra) = self.rx.recv_timeout(rem) else {
                            break;
                        };
                        if extra.get("method").is_some()
                            && let Some(req_id) = extra.get("id")
                        {
                            let req_method = extra.get("method").and_then(|m| m.as_str()).unwrap_or("");
                            let _ = write_lsp_frame(
                                &mut self.stdin,
                                &serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "result": serde_json::Value::Null
                                }),
                            );
                            if req_method == "workspace/inlayHint/refresh"
                                || req_method == "workspace/semanticTokens/refresh"
                            {
                                break;
                            }
                            continue;
                        }
                        if extra.get("method").and_then(|m| m.as_str())
                            == Some("textDocument/publishDiagnostics")
                            && extra
                                .get("params")
                                .and_then(|p| p.get("version"))
                                .and_then(|v| v.as_u64())
                                == Some(ver)
                        {
                            latest_diags = extra
                                .get("params")
                                .and_then(|p| p.get("diagnostics"))
                                .and_then(|d| d.as_array())
                                .cloned()
                                .unwrap_or_default();
                            drain_until = Instant::now() + Duration::from_millis(20);
                        }
                    }
                    break;
                }
            }
        }

        let _ = self.next_id;
        Ok(ServerEvalOutcome::Ndjson(lsp_diags_to_ndjson(&latest_diags)))
    }

    fn query_plain_goals_at(
        &mut self,
        line_0based: usize,
        col_0based: usize,
        deadline: Option<Instant>,
    ) -> Result<Vec<String>, ProverError> {
        let req_id = self.next_id;
        self.next_id += 1;
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "$/lean/plainGoal",
            "params": {
                "textDocument": { "uri": self.uri },
                "position": { "line": line_0based, "character": col_0based }
            }
        });
        write_lsp_frame(&mut self.stdin, &msg)?;
        let max_wait = deadline.unwrap_or_else(|| Instant::now() + Duration::from_secs(10));
        loop {
            let now = Instant::now();
            if now >= max_wait {
                return Ok(Vec::new());
            }
            let wait_for = max_wait.saturating_duration_since(now);
            let Ok(frame) = self.rx.recv_timeout(wait_for) else {
                return Ok(Vec::new());
            };
            if frame.get("method").is_some()
                && let Some(srv_id) = frame.get("id")
            {
                let _ = write_lsp_frame(
                    &mut self.stdin,
                    &serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": srv_id,
                        "result": serde_json::Value::Null
                    }),
                );
                continue;
            }
            if frame.get("id").and_then(|v| v.as_u64()) == Some(req_id) {
                let goals = frame
                    .get("result")
                    .and_then(|r| r.get("goals"))
                    .and_then(|g| g.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                return Ok(goals);
            }
        }
    }
}

fn lsp_diags_to_ndjson(diags: &[serde_json::Value]) -> String {
    let mut lines = Vec::with_capacity(diags.len());
    for d in diags {
        let sev = match d.get("severity").and_then(|s| s.as_u64()).unwrap_or(1) {
            1 => "error",
            2 => "warning",
            _ => "information",
        };
        let range = d.get("fullRange").or_else(|| d.get("range"));
        let sl = range
            .and_then(|r| r.get("start"))
            .and_then(|p| p.get("line"))
            .and_then(|l| l.as_u64())
            .unwrap_or(0)
            + 1;
        let sc = range
            .and_then(|r| r.get("start"))
            .and_then(|p| p.get("character"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0);
        let el = range
            .and_then(|r| r.get("end"))
            .and_then(|p| p.get("line"))
            .and_then(|l| l.as_u64())
            .unwrap_or(0)
            + 1;
        let ec = range
            .and_then(|r| r.get("end"))
            .and_then(|p| p.get("character"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0);
        let data = d.get("message").and_then(|m| m.as_str()).unwrap_or("");
        let obj = serde_json::json!({
            "severity": sev,
            "pos": { "line": sl, "column": sc },
            "endPos": { "line": el, "column": ec },
            "data": data,
        });
        lines.push(obj.to_string());
    }
    lines.join("\n")
}

thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static PERSISTENT_LEAN_SERVER: RefCell<Option<LeanServerSession>> = const { RefCell::new(None) };
    #[allow(clippy::missing_const_for_thread_local)]
    static SESSION_UNKNOWN_CONSTANTS: RefCell<std::collections::HashSet<String>> =
        RefCell::new(std::collections::HashSet::new());
}

fn extract_unknown_constants_from_diag(diag: &str) -> Vec<String> {
    let mut found = Vec::new();
    for marker in ["Unknown constant ", "Unknown identifier "] {
        let mut rest = diag;
        while let Some(pos) = rest.find(marker) {
            let after = &rest[pos + marker.len()..];
            let token = after
                .trim_start()
                .trim_start_matches(['`', '\'', '"'])
                .split(|c: char| c.is_whitespace() || matches!(c, '`' | '\'' | '"' | ',' | ';' | ':' | ')' | ']'))
                .next()
                .unwrap_or("")
                .trim();
            if !token.is_empty() && !found.iter().any(|x| x == token) {
                found.push(token.to_string());
            }
            rest = after;
        }
    }
    found
}

pub(crate) fn record_unknown_constants_from_diag(diag: &str) -> Vec<String> {
    let found = extract_unknown_constants_from_diag(diag);
    if !found.is_empty() {
        SESSION_UNKNOWN_CONSTANTS.with(|cell| {
            let mut set = cell.borrow_mut();
            for name in &found {
                set.insert(name.clone());
            }
        });
    }
    found
}

pub(crate) fn is_unknown_constant(name: &str) -> bool {
    SESSION_UNKNOWN_CONSTANTS.with(|cell| cell.borrow().contains(name))
}

fn strip_unknown_constants_from_script(
    script: &[String],
    bad: &std::collections::HashSet<String>,
) -> Option<Vec<String>> {
    if bad.is_empty() {
        return None;
    }
    let mut changed = false;
    let mut out = Vec::with_capacity(script.len());
    for line in script {
        let mut cur = line.clone();
        if let (Some(open), Some(close)) = (cur.find('['), cur.rfind(']'))
            && open < close
        {
            let before = cur[..open].trim_end();
            let inside = &cur[open + 1..close];
            let after = &cur[close + 1..];
            let items: Vec<&str> = inside
                .split(',')
                .map(str::trim)
                .filter(|item| {
                    let clean = item.trim_start_matches('←').trim();
                    !clean.is_empty() && !bad.contains(clean)
                })
                .collect();
            let orig_count = inside.split(',').filter(|s| !s.trim().is_empty()).count();
            if items.len() < orig_count {
                changed = true;
                if items.is_empty() {
                    if before.ends_with("rw") || before.ends_with("erw") {
                        return None;
                    }
                    cur = format!("{before}{after}");
                } else {
                    cur = format!("{before} [{}]{after}", items.join(", "));
                }
            }
        }
        let trimmed = cur.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
    }
    if changed && !out.is_empty() {
        Some(out)
    } else {
        None
    }
}

fn eval_via_lean_server(
    src: &str,
    lake_root: Option<&Path>,
    deadline: Option<Instant>,
) -> Result<ServerEvalOutcome, ProverError> {
    if let Some(dl) = deadline
        && Instant::now() >= dl
    {
        return Ok(ServerEvalOutcome::TimedOut);
    }
    PERSISTENT_LEAN_SERVER.with(|cell| {
        let mut slot = cell.borrow_mut();
        let target_root = lake_root.map(normalize_lake_root);
        let needs_spawn = match slot.as_mut() {
            Some(sess) => {
                sess.lake_root != target_root
                    || matches!(sess.child.try_wait(), Ok(Some(_)) | Err(_))
            }
            None => true,
        };
        if needs_spawn {
            *slot = None;
            match LeanServerSession::spawn(lake_root, deadline) {
                Ok(s) => *slot = Some(s),
                Err(_) if deadline.is_some_and(|dl| Instant::now() >= dl) => {
                    return Ok(ServerEvalOutcome::TimedOut);
                }
                Err(e) => return Err(e),
            }
        }
        let outcome = match slot.as_mut().unwrap().eval_document(src, deadline) {
            Ok(out) => out,
            Err(_) => {
                *slot = None;
                if let Some(dl) = deadline
                    && Instant::now() >= dl
                {
                    return Ok(ServerEvalOutcome::TimedOut);
                }
                *slot = Some(LeanServerSession::spawn(lake_root, deadline)?);
                slot.as_mut().unwrap().eval_document(src, deadline)?
            }
        };
        if matches!(outcome, ServerEvalOutcome::TimedOut) {
            *slot = None;
        }
        Ok(outcome)
    })
}

/// Evaluate `partial_script` followed by `sorry` in `thm_name` and query `$/lean/plainGoal`
/// anchored directly on the trailing `sorry` token.
pub(crate) fn query_plain_goals_for_script(
    src: &str,
    thm_name: &str,
    partial_script: &[String],
    lake_root: Option<&Path>,
    deadline: Option<Instant>,
) -> Result<Vec<String>, ProverError> {
    let mut lines: Vec<String> = partial_script
        .iter()
        .map(|l| format!("  {l}"))
        .collect();
    lines.push("  sorry".to_string());
    let body = lines.join("\n");
    let probe_src = replace_theorem_sorry(src, thm_name, &body);
    let (span_start, span_end) = target_theorem_line_span(&probe_src, thm_name);
    let mut target_pos: Option<(usize, usize)> = None;
    for (idx, line) in probe_src.lines().enumerate() {
        let lnum = idx + 1;
        if lnum >= span_start && lnum <= span_end
            && let Some(col) = line.rfind("sorry")
        {
            target_pos = Some((idx, col));
        }
    }
    let Some((line_0, col_0)) = target_pos else {
        return Ok(Vec::new());
    };
    if let Some(dl) = deadline
        && Instant::now() >= dl
    {
        return Ok(Vec::new());
    }
    PERSISTENT_LEAN_SERVER.with(|cell| {
        let mut slot = cell.borrow_mut();
        let target_root = lake_root.map(normalize_lake_root);
        let needs_spawn = match slot.as_mut() {
            Some(sess) => {
                sess.lake_root != target_root
                    || matches!(sess.child.try_wait(), Ok(Some(_)) | Err(_))
            }
            None => true,
        };
        if needs_spawn {
            *slot = None;
            match LeanServerSession::spawn(lake_root, deadline) {
                Ok(s) => *slot = Some(s),
                Err(_) if deadline.is_some_and(|dl| Instant::now() >= dl) => {
                    return Ok(Vec::new());
                }
                Err(e) => return Err(e),
            }
        }
        let sess = slot.as_mut().unwrap();
        match sess.eval_document(&probe_src, deadline)? {
            ServerEvalOutcome::TimedOut => {
                *slot = None;
                Ok(Vec::new())
            }
            ServerEvalOutcome::Ndjson(_) => sess.query_plain_goals_at(line_0, col_0, deadline),
        }
    })
}

/// Late OOD fallback: run `exact?` under `deadline`, extract the `"Try this: ..."` term,
/// and kernel-verify the concrete replacement (never accepting `exact?` itself).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_exact_search_fallback(
    src: &str,
    thm: &DiscoveredTheorem,
    full_name: &str,
    lake_root: Option<&Path>,
    prune_unused_simp: bool,
    strict_lint: bool,
    deadline: Option<Instant>,
) -> Result<Option<(String, Vec<String>)>, ProverError> {
    if let Some(dl) = deadline
        && Instant::now() >= dl
    {
        return Ok(None);
    }
    let probe_src = replace_theorem_sorry(src, &thm.name, "  exact?");
    let ndjson = match eval_via_lean_server(&probe_src, lake_root, deadline)? {
        ServerEvalOutcome::Ndjson(s) => s,
        ServerEvalOutcome::TimedOut => return Ok(None),
    };
    for line in ndjson.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let data = v.get("data").and_then(|d| d.as_str()).unwrap_or("");
            if let Some(after_try) = data.split("Try this:").nth(1) {
                let cleaned = after_try
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .strip_prefix("[apply]")
                    .map(str::trim)
                    .unwrap_or_else(|| after_try.lines().next().unwrap_or("").trim());
                if !cleaned.is_empty() && !cleaned.contains("exact?") && !cleaned.contains("sorry") {
                    let concrete = vec![cleaned.to_string()];
                    if let Ok(Ok(verified)) = audit_and_prune_unused_simp_args(
                        src,
                        &thm.name,
                        full_name,
                        &concrete,
                        lake_root,
                        prune_unused_simp,
                        strict_lint,
                    ) {
                        return Ok(Some(verified));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// Audit a single candidate script by replacing `sorry` and running `#print axioms` through `verify_allowed_axioms`.
pub fn audit_single_candidate(
    src: &str,
    thm_name: &str,
    full_name: &str,
    script: &[String],
    lake_root: Option<&Path>,
) -> Result<Result<String, String>, ProverError> {
    let (res, _, _, _) =
        audit_single_candidate_raw(src, thm_name, full_name, script, lake_root, false)?;
    Ok(res)
}

fn is_decl_boundary_line(trimmed: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "theorem ",
        "lemma ",
        "private theorem ",
        "private lemma ",
        "protected theorem ",
        "protected lemma ",
        "def ",
        "private def ",
        "inductive ",
        "structure ",
        "class ",
        "instance ",
        "namespace ",
        "section ",
        "end ",
        "/--",
        "/-!",
        "#print axioms ",
    ];
    PREFIXES.iter().any(|p| trimmed.starts_with(p))
}

fn target_theorem_line_span(audit_src: &str, thm_name: &str) -> (usize, usize) {
    let lines: Vec<&str> = audit_src.lines().collect();
    let mut start_line: Option<usize> = None;
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        for kw in ["theorem ", "lemma "] {
            if let Some(pos) = trimmed.find(kw) {
                let after_kw = trimmed[pos + kw.len()..].trim_start();
                if let Some(rest) = after_kw.strip_prefix(thm_name) {
                    let boundary_ok = rest.is_empty()
                        || rest.starts_with(|c: char| {
                            c.is_whitespace() || matches!(c, ':' | '(' | '{' | '[')
                        });
                    if boundary_ok {
                        start_line = Some(idx + 1);
                        break;
                    }
                }
            }
        }
        if start_line.is_some() {
            break;
        }
    }
    let Some(start) = start_line else {
        return (1, lines.len().max(1));
    };
    let mut end = lines.len();
    for (idx, line) in lines.iter().enumerate().skip(start) {
        if is_decl_boundary_line(line.trim_start()) {
            end = idx;
            break;
        }
    }
    (start, end.max(start))
}

fn audit_single_candidate_raw(
    src: &str,
    thm_name: &str,
    full_name: &str,
    script: &[String],
    lake_root: Option<&Path>,
    strict_lint: bool,
) -> Result<(Result<String, String>, String, String, String), ProverError> {
    let body = script
        .iter()
        .map(|l| format!("  {l}"))
        .collect::<Vec<_>>()
        .join("\n");
    let candidate_src = replace_theorem_sorry(src, thm_name, &body);
    let audit = format!("{candidate_src}\n\n#print axioms {full_name}\n");
    let stdout = match eval_via_lean_server(&audit, lake_root, None)? {
        ServerEvalOutcome::Ndjson(s) => s,
        ServerEvalOutcome::TimedOut => return Ok((Err("audit timed out".to_string()), body, audit, String::new())),
    };

    let (thm_start_line, thm_end_line) = target_theorem_line_span(&audit, thm_name);
    let mut has_error = false;
    let mut illegal_axiom: Option<String> = None;
    let mut first_error_msg: Option<String> = None;

    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let sev = v.get("severity").and_then(|s| s.as_str()).unwrap_or("");
            let data = v.get("data").and_then(|d| d.as_str()).unwrap_or("");
            if sev == "error" {
                has_error = true;
                if first_error_msg.is_none() && !data.is_empty() {
                    first_error_msg = Some(data.to_string());
                }
            } else if strict_lint && sev == "warning" {
                let line_num = v
                    .get("pos")
                    .and_then(|p| p.get("line"))
                    .and_then(|l| l.as_u64())
                    .map(|l| l as usize);
                let in_target_span =
                    line_num.is_some_and(|l| (thm_start_line..=thm_end_line).contains(&l));
                if in_target_span && !data.is_empty() {
                    has_error = true;
                    if first_error_msg.is_none() {
                        first_error_msg = Some(format!("strict-lint warning in {thm_name}: {data}"));
                    }
                }
            }
            if (data.contains("depends on axioms:")
                || data.contains("does not depend on any axioms"))
                && let Err(e) = lean_repl::verify_allowed_axioms(data)
            {
                illegal_axiom = Some(e.to_string());
            }
        }
    }

    if !has_error && illegal_axiom.is_none() {
        Ok((Ok(candidate_src), body, audit, stdout))
    } else {
        let diag = illegal_axiom
            .or(first_error_msg)
            .or_else(|| {
                stdout
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "kernel rejected".to_string());
        Ok((Err(diag), body, audit, stdout))
    }
}

fn char_col_to_byte(line: &str, col: usize) -> usize {
    line.chars().take(col).map(char::len_utf8).sum()
}

fn line_col_to_byte_offset(src: &str, line_1based: usize, col_0based: usize) -> Option<usize> {
    if line_1based == 0 {
        return None;
    }
    let mut offset = 0usize;
    for (idx, raw_line) in src.split_inclusive('\n').enumerate() {
        if idx + 1 == line_1based {
            let line_no_nl = raw_line.strip_suffix('\n').unwrap_or(raw_line);
            return Some(offset + char_col_to_byte(line_no_nl, col_0based));
        }
        offset += raw_line.len();
    }
    None
}

#[derive(Debug, Clone)]
struct UnusedSimpOccurrence {
    abs_start: Option<usize>,
    ident: String,
}

fn collect_unused_simp_occurrences(audit_src: &str, stdout: &str) -> Vec<UnusedSimpOccurrence> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v.get("severity").and_then(|s| s.as_str()) != Some("warning") {
            continue;
        }
        let data = v.get("data").and_then(|d| d.as_str()).unwrap_or("");
        if !data.contains("This simp argument is unused") && !data.contains("unusedSimpArgs") {
            continue;
        }
        let span_info = v.get("pos").and_then(|p| {
            let pl = p.get("line")?.as_u64()? as usize;
            let pc = p.get("column")?.as_u64()? as usize;
            let ep = v.get("endPos")?;
            let el = ep.get("line")?.as_u64()? as usize;
            let ec = ep.get("column")?.as_u64()? as usize;
            let abs_start = line_col_to_byte_offset(audit_src, pl, pc)?;
            let abs_end = line_col_to_byte_offset(audit_src, el, ec)?;
            if abs_start < abs_end && abs_end <= audit_src.len() {
                Some((abs_start, audit_src[abs_start..abs_end].trim().to_string()))
            } else {
                None
            }
        });
        let (abs_start, ident) = match span_info {
            Some((s, id)) if !id.is_empty() => (Some(s), id),
            _ => {
                let fallback = data.find('`').and_then(|s| {
                    let rest = &data[s + 1..];
                    let e = rest.find('`')?;
                    Some(rest[..e].trim().to_string())
                });
                match fallback {
                    Some(id) if !id.is_empty() => (None, id),
                    _ => continue,
                }
            }
        };
        out.push(UnusedSimpOccurrence { abs_start, ident });
    }
    out
}

fn prune_unused_simp_args_by_span(
    body: &str,
    audit_src: &str,
    occurrences: &[UnusedSimpOccurrence],
) -> String {
    let body_offset = audit_src.find(body);
    let mut out = body.to_string();

    // Collect all `simp [` and `simp_all [` bracket ranges in `body` from back to front so
    // replacing a range never shifts earlier offsets.
    let mut brackets: Vec<(usize, usize, &'static str)> = Vec::new();
    for kw in ["simp_all [", "simp only [", "dsimp only [", "dsimp [", "simp ["] {
        let mut search_from = 0usize;
        while let Some(rel) = body[search_from..].find(kw) {
            let kw_start = search_from + rel;
            let open_idx = kw_start + kw.len() - 1;
            if let Some(close_rel) = body[open_idx + 1..].find(']') {
                let close_idx = open_idx + 1 + close_rel;
                brackets.push((kw_start, close_idx, kw));
                search_from = close_idx + 1;
            } else {
                break;
            }
        }
    }
    brackets.sort_by_key(|b| std::cmp::Reverse(b.0));

    for (kw_start, close_idx, kw) in brackets {
        let open_idx = kw_start + kw.len() - 1;
        let mut local_unused: Vec<&str> = Vec::new();
        for occ in occurrences {
            let matches_here = match (occ.abs_start, body_offset) {
                (Some(abs), Some(base)) if abs >= base => {
                    let rel = abs - base;
                    rel > open_idx && rel < close_idx
                }
                _ => true,
            };
            if matches_here && !local_unused.contains(&occ.ident.as_str()) {
                local_unused.push(occ.ident.as_str());
            }
        }
        if local_unused.is_empty() {
            continue;
        }
        let inner = &out[open_idx + 1..close_idx];
        let kept: Vec<&str> = inner
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty() && !local_unused.contains(item))
            .collect();
        let kw_name = kw.trim_end_matches(" [");
        let replacement = if kept.is_empty() {
            kw_name.to_string()
        } else {
            format!("{kw_name} [{}]", kept.join(", "))
        };
        out.replace_range(kw_start..=close_idx, &replacement);
    }

    out
}

fn audit_and_prune_unused_simp_args(
    src: &str,
    thm_name: &str,
    full_name: &str,
    script: &[String],
    lake_root: Option<&Path>,
    prune_unused_simp: bool,
    strict_lint: bool,
) -> Result<Result<(String, Vec<String>), String>, ProverError> {
    let (res, mut cur_body, mut cur_audit, mut cur_stdout) =
        audit_single_candidate_raw(src, thm_name, full_name, script, lake_root, false)?;
    let Ok(mut cur_src) = res else {
        return Ok(Err(res.unwrap_err()));
    };
    let mut cur_script = script.to_vec();

    let has_simp_list = cur_script.iter().any(|l| {
        l.contains("simp [")
            || l.contains("simp_all [")
            || l.contains("simp only [")
            || l.contains("dsimp only [")
            || l.contains("dsimp [")
    });
    if prune_unused_simp && has_simp_list {
        for _ in 0..4 {
            let occurrences = collect_unused_simp_occurrences(&cur_audit, &cur_stdout);
            if occurrences.is_empty() {
                break;
            }
            let pruned_body = prune_unused_simp_args_by_span(&cur_body, &cur_audit, &occurrences);
            let pruned_script: Vec<String> = pruned_body
                .lines()
                .map(|l| l.strip_prefix("  ").unwrap_or(l).to_string())
                .collect();
            if pruned_script == cur_script {
                break;
            }
            let (re_res, re_body, re_audit, re_stdout) = audit_single_candidate_raw(
                src,
                thm_name,
                full_name,
                &pruned_script,
                lake_root,
                false,
            )?;
            let Ok(pruned_src) = re_res else {
                break;
            };
            let remaining = collect_unused_simp_occurrences(&re_audit, &re_stdout);
            if remaining.len() >= occurrences.len() {
                break;
            }
            cur_src = pruned_src;
            cur_body = re_body;
            cur_audit = re_audit;
            cur_stdout = re_stdout;
            cur_script = pruned_script;
            if remaining.is_empty() {
                break;
            }
        }
    }

    if strict_lint {
        let (strict_res, _, _, _) =
            audit_single_candidate_raw(src, thm_name, full_name, &cur_script, lake_root, true)?;
        return Ok(strict_res.map(|s| (s, cur_script)));
    }

    Ok(Ok((cur_src, cur_script)))
}

fn find_matching_block_comment_start(s: &str) -> Option<usize> {
    if !s.ends_with("-/") {
        return None;
    }
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    let mut i = bytes.len();
    while i >= 2 {
        if bytes[i - 2] == b'-' && bytes[i - 1] == b'/' {
            depth += 1;
            i -= 2;
        } else if bytes[i - 2] == b'/' && bytes[i - 1] == b'-' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(i - 2);
            }
            i -= 2;
        } else {
            i -= 1;
        }
    }
    None
}

fn strip_trailing_docstrings_and_attrs(prefix: &str) -> &str {
    let mut s = prefix.trim_end();
    loop {
        if let Some(start) = find_matching_block_comment_start(s) {
            s = s[..start].trim_end();
            continue;
        }
        if let Some(last_nl) = s.rfind('\n') {
            let last_line = s[last_nl + 1..].trim();
            if last_line.starts_with("@[") && last_line.ends_with(']') {
                s = s[..last_nl].trim_end();
                continue;
            }
        } else {
            let trimmed = s.trim();
            if trimmed.starts_with("@[") && trimmed.ends_with(']') {
                s = "";
                continue;
            }
        }
        break;
    }
    s
}

fn run_lean_src_messages(
    src: &str,
    lake_root: Option<&std::path::Path>,
) -> Result<Vec<serde_json::Value>, ProverError> {
    let stdout = match eval_via_lean_server(src, lake_root, None)? {
        ServerEvalOutcome::Ndjson(s) => s,
        ServerEvalOutcome::TimedOut => String::new(),
    };
    let mut msgs = Vec::new();
    for line in stdout.lines() {
        if let Ok(m) = serde_json::from_str::<serde_json::Value>(line.trim()) {
            msgs.push(m);
        }
    }
    Ok(msgs)
}

/// Pre-flight kernel probe classifying `(BuildStatus, SanityStatus, Option<winning_script_or_diag>)`
/// before entering ONNX / VDB / tree search.
pub fn probe_preflight_build_and_sanity(
    src: &str,
    thm: &DiscoveredTheorem,
    _full_name: &str,
    lake_root: Option<&std::path::Path>,
) -> Result<(crate::BuildStatus, crate::SanityStatus, Option<String>), ProverError> {
    let triv_tactic = "first | rfl | decide | trivial";
    let triv_src = crate::replace_theorem_sorry(src, &thm.name, &format!("  {triv_tactic}"));
    let triv_msgs = run_lean_src_messages(&triv_src, lake_root)?;
    let (span_start, span_end) = target_theorem_line_span(&triv_src, &thm.name);

    let mut has_error = false;
    let mut external_err: Option<String> = None;
    for msg in &triv_msgs {
        if msg.get("severity").and_then(|s| s.as_str()) == Some("error") {
            has_error = true;
            let line = msg
                .get("pos")
                .and_then(|p| p.get("line"))
                .and_then(|l| l.as_u64())
                .unwrap_or(0) as usize;
            let data = msg.get("data").and_then(|d| d.as_str()).unwrap_or("").trim();
            if (line == 0 || line < span_start || line > span_end) && external_err.is_none() {
                external_err = Some(format!("compile error: {data}"));
            }
        }
    }
    if let Some(err) = external_err {
        return Ok((crate::BuildStatus::CompileError, crate::SanityStatus::ValidOpen, Some(err)));
    }
    if !has_error {
        return Ok((
            crate::BuildStatus::Ok,
            crate::SanityStatus::Trivial,
            Some(triv_tactic.to_string()),
        ));
    }

    let false_src = crate::replace_theorem_sorry(
        src,
        &thm.name,
        "  intros; try simp_all; trace_state; sorry",
    );
    let false_msgs = run_lean_src_messages(&false_src, lake_root)?;
    for msg in &false_msgs {
        let d = msg.get("data").and_then(|s| s.as_str()).unwrap_or("");
        if d.contains("⊢ False") || d.contains("⊢ true = false") || d.contains("⊢ false = true") {
            return Ok((
                crate::BuildStatus::Ok,
                crate::SanityStatus::FalseSpec,
                Some(format!("false spec: {}", d.trim())),
            ));
        }
    }

    Ok((crate::BuildStatus::Ok, crate::SanityStatus::ValidOpen, None))
}

/// Evaluate candidate scripts in batches of 8 with per-probe `set_option maxHeartbeats <N> in` isolation.
pub fn verify_candidates_batched_with_heartbeats(
    src: &str,
    thm: &DiscoveredTheorem,
    full_name: &str,
    candidates: &[Vec<String>],
    lake_root: Option<&Path>,
    max_heartbeats: usize,
) -> Result<BatchedVerifyResult, ProverError> {
    verify_candidates_batched_with_heartbeats_and_ui(
        src,
        thm,
        full_name,
        candidates,
        lake_root,
        max_heartbeats,
        &UiHandle::noop(),
        0,
    )
}

/// Evaluate candidate scripts in batches of 8, streaming `lake env lean --json` diagnostics live to `ui`.
#[allow(clippy::too_many_arguments)]
pub fn verify_candidates_batched_with_heartbeats_and_ui(
    src: &str,
    thm: &DiscoveredTheorem,
    full_name: &str,
    candidates: &[Vec<String>],
    lake_root: Option<&Path>,
    max_heartbeats: usize,
    ui: &UiHandle,
    branch_offset: usize,
) -> Result<BatchedVerifyResult, ProverError> {
    verify_candidates_batched_with_options_and_ui(
        src,
        thm,
        full_name,
        candidates,
        lake_root,
        max_heartbeats,
        ui,
        branch_offset,
        false,
        false,
    )
}

/// Evaluate candidate scripts in batches with optional `--write`-gated `unusedSimpArgs` pruning and `--strict-lint`.
#[allow(clippy::too_many_arguments)]
pub fn verify_candidates_batched_with_options_and_ui(
    src: &str,
    thm: &DiscoveredTheorem,
    full_name: &str,
    candidates: &[Vec<String>],
    lake_root: Option<&Path>,
    max_heartbeats: usize,
    ui: &UiHandle,
    branch_offset: usize,
    prune_unused_simp: bool,
    strict_lint: bool,
) -> Result<BatchedVerifyResult, ProverError> {
    verify_candidates_batched_with_deadline_and_ui(
        src,
        thm,
        full_name,
        candidates,
        lake_root,
        max_heartbeats,
        ui,
        branch_offset,
        prune_unused_simp,
        strict_lint,
        None,
    )
}

/// Evaluate candidate scripts in batches with optional `unusedSimpArgs` pruning, `--strict-lint`, and a hard wall-clock `deadline`.
#[allow(clippy::too_many_arguments)]
pub fn verify_candidates_batched_with_deadline_and_ui(
    src: &str,
    thm: &DiscoveredTheorem,
    full_name: &str,
    candidates: &[Vec<String>],
    lake_root: Option<&Path>,
    max_heartbeats: usize,
    ui: &UiHandle,
    branch_offset: usize,
    prune_unused_simp: bool,
    strict_lint: bool,
    deadline: Option<Instant>,
) -> Result<BatchedVerifyResult, ProverError> {
    let mut last_diag = String::from("model produced no valid kernel-verified tactic");
    let mut unsolved_diags: Vec<(Vec<String>, String)> = Vec::new();
    if candidates.is_empty() {
        return Ok((None, last_diag, unsolved_diags));
    }

    let needle = format!("theorem {}", thm.name);
    let Some(thm_pos) = src.find(&needle) else {
        for (idx, script) in candidates.iter().enumerate() {
            let branch_id = branch_offset + idx + 1;
            let preview = script.join("; ");
            ui.send(UiEvent::BranchSpawned {
                id: branch_id,
                tag: "Probe".to_string(),
                preview: preview.clone(),
            });
            ui.send(UiEvent::BranchUpdated {
                id: branch_id,
                status: BranchStatus::Running,
                elapsed_ms: 0,
            });
            let t0 = Instant::now();
            match audit_and_prune_unused_simp_args(
                src,
                &thm.name,
                full_name,
                script,
                lake_root,
                prune_unused_simp,
                strict_lint,
            )? {
                Ok((cand_src, final_script)) => {
                    ui.send(UiEvent::BranchUpdated {
                        id: branch_id,
                        status: BranchStatus::Verified {
                            tactic: final_script.join("; "),
                        },
                        elapsed_ms: t0.elapsed().as_millis() as u64,
                    });
                    return Ok((
                        Some((cand_src, final_script)),
                        String::new(),
                        unsolved_diags,
                    ));
                }
                Err(d) => {
                    ui.send(UiEvent::BranchUpdated {
                        id: branch_id,
                        status: BranchStatus::Failed {
                            reason: d.lines().next().unwrap_or("failed").to_string(),
                        },
                        elapsed_ms: t0.elapsed().as_millis() as u64,
                    });
                    if d.contains("unsolved goals") {
                        unsolved_diags.push((script.clone(), d.clone()));
                    }
                    last_diag = d;
                }
            }
        }
        return Ok((None, last_diag, unsolved_diags));
    };

    let prefix = strip_trailing_docstrings_and_attrs(&src[..thm_pos]);
    let rest = &src[thm_pos..];
    let Some(by_rel) = rest.find(":= by") else {
        return Ok((None, last_diag, unsolved_diags));
    };
    let after_name = &rest[needle.len()..by_rel];
    let hb = max_heartbeats.max(10_000);

    let batch_size: usize = if lake_root.is_some() { 16 } else { 8 };
    for (chunk_idx, chunk) in candidates.chunks(batch_size).enumerate() {
        if let Some(dl) = deadline
            && Instant::now() >= dl
        {
            last_diag = "ood escalation budget exceeded".to_string();
            break;
        }
        let chunk_base_id = branch_offset + chunk_idx * batch_size;
        let batch_start = Instant::now();
        let mut batch_src = String::from(prefix);
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(chunk.len());
        let mut cur_line = batch_src.lines().count()
            + if batch_src.ends_with('\n') || batch_src.is_empty() {
                1
            } else {
                2
            };
        if !batch_src.ends_with('\n') && !batch_src.is_empty() {
            batch_src.push('\n');
        }

        for (i, script) in chunk.iter().enumerate() {
            let start_line = cur_line;
            let body = script
                .iter()
                .map(|l| format!("  {l}"))
                .collect::<Vec<_>>()
                .join("\n");
            let decl = format!(
                "set_option maxHeartbeats {hb} in\nprivate theorem __probe_{i}{after_name}:= by\n{body}\n"
            );
            cur_line += decl.lines().count();
            ranges.push((start_line, cur_line.saturating_sub(1)));
            batch_src.push_str(&decl);

            let branch_id = chunk_base_id + i + 1;
            let preview = script.join("; ");
            ui.send(UiEvent::BranchSpawned {
                id: branch_id,
                tag: "Batch".to_string(),
                preview,
            });
            ui.send(UiEvent::BranchUpdated {
                id: branch_id,
                status: BranchStatus::Running,
                elapsed_ms: 0,
            });
        }

        let ndjson = match eval_via_lean_server(&batch_src, lake_root, deadline)? {
            ServerEvalOutcome::Ndjson(s) => s,
            ServerEvalOutcome::TimedOut => {
                last_diag = "ood escalation budget exceeded".to_string();
                break;
            }
        };
        let mut cand_errors: Vec<Option<String>> = vec![None; chunk.len()];
        let mut has_unmapped_error = false;

        for line in ndjson.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let sev = v.get("severity").and_then(|s| s.as_str()).unwrap_or("");
                if sev != "error" {
                    continue;
                }
                let data = v
                    .get("data")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
                let err_line = v
                    .get("pos")
                    .and_then(|p| p.get("line"))
                    .and_then(|l| l.as_u64())
                    .map(|l| l as usize);
                if let Some(el) = err_line
                    && let Some(idx) = ranges.iter().position(|&(s, e)| el >= s && el <= e)
                {
                    if cand_errors[idx].is_none() {
                        record_unknown_constants_from_diag(&data);
                        ui.send(UiEvent::BranchUpdated {
                            id: chunk_base_id + idx + 1,
                            status: BranchStatus::Failed {
                                reason: data.lines().next().unwrap_or("error").to_string(),
                            },
                            elapsed_ms: batch_start.elapsed().as_millis() as u64,
                        });
                        cand_errors[idx] = Some(data);
                    }
                } else {
                    has_unmapped_error = true;
                    last_diag = data;
                }
            }
        }

        // Retry any candidates poisoned by unimported VDB constants after stripping blacklisted names
        let mut retry_scripts: Vec<Vec<String>> = Vec::new();
        SESSION_UNKNOWN_CONSTANTS.with(|cell| {
            let bad = cell.borrow();
            if !bad.is_empty() {
                for (i, err) in cand_errors.iter().enumerate() {
                    if let Some(msg) = err
                        && (msg.contains("Unknown constant") || msg.contains("Unknown identifier"))
                        && let Some(stripped) = strip_unknown_constants_from_script(&chunk[i], &bad)
                        && !retry_scripts.contains(&stripped)
                    {
                        retry_scripts.push(stripped);
                    }
                }
            }
        });
        if !retry_scripts.is_empty() && deadline.is_none_or(|dl| Instant::now() < dl) {
            let mut retry_src = String::from(prefix);
            if !retry_src.ends_with('\n') && !retry_src.is_empty() {
                retry_src.push('\n');
            }
            let mut r_line = retry_src.lines().count() + 1;
            let mut r_ranges = Vec::with_capacity(retry_scripts.len());
            for (i, r_script) in retry_scripts.iter().enumerate() {
                let start_l = r_line;
                let body = r_script
                    .iter()
                    .map(|l| format!("  {l}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                let decl = format!(
                    "set_option maxHeartbeats {hb} in\nprivate theorem __retry_{i}{after_name}:= by\n{body}\n"
                );
                r_line += decl.lines().count();
                r_ranges.push((start_l, r_line.saturating_sub(1)));
                retry_src.push_str(&decl);
            }
            if let Ok(ServerEvalOutcome::Ndjson(r_ndjson)) =
                eval_via_lean_server(&retry_src, lake_root, deadline)
            {
                let mut r_errs: Vec<bool> = vec![false; retry_scripts.len()];
                for rline in r_ndjson.lines() {
                    if let Ok(rv) = serde_json::from_str::<serde_json::Value>(rline)
                        && rv.get("severity").and_then(|s| s.as_str()) == Some("error")
                        && let Some(el) = rv
                            .get("pos")
                            .and_then(|p| p.get("line"))
                            .and_then(|l| l.as_u64())
                            .map(|l| l as usize)
                        && let Some(ridx) = r_ranges.iter().position(|&(s, e)| el >= s && el <= e)
                    {
                        r_errs[ridx] = true;
                    }
                }
                for (i, has_err) in r_errs.iter().enumerate() {
                    if !has_err
                        && let Ok(Ok((cand_src, final_script))) = audit_and_prune_unused_simp_args(
                            src,
                            &thm.name,
                            full_name,
                            &retry_scripts[i],
                            lake_root,
                            prune_unused_simp,
                            strict_lint,
                        )
                    {
                        return Ok((Some((cand_src, final_script)), String::new(), unsolved_diags));
                    }
                }
            }
        }

        if !has_unmapped_error {
            for (i, err) in cand_errors.iter().enumerate() {
                if err.is_none() {
                    let script = &chunk[i];
                    match audit_and_prune_unused_simp_args(
                        src,
                        &thm.name,
                        full_name,
                        script,
                        lake_root,
                        prune_unused_simp,
                        strict_lint,
                    )? {
                        Ok((cand_src, final_script)) => {
                            let elapsed_ms = batch_start.elapsed().as_millis() as u64;
                            ui.send(UiEvent::BranchUpdated {
                                id: chunk_base_id + i + 1,
                                status: BranchStatus::Verified {
                                    tactic: final_script.join("; "),
                                },
                                elapsed_ms,
                            });
                            for (j, other_err) in cand_errors.iter().enumerate() {
                                if j != i && other_err.is_none() {
                                    ui.send(UiEvent::BranchUpdated {
                                        id: chunk_base_id + j + 1,
                                        status: BranchStatus::Canceled,
                                        elapsed_ms,
                                    });
                                }
                            }
                            return Ok((
                                Some((cand_src, final_script)),
                                String::new(),
                                unsolved_diags,
                            ));
                        }
                        Err(d) => {
                            ui.send(UiEvent::BranchUpdated {
                                id: chunk_base_id + i + 1,
                                status: BranchStatus::Failed {
                                    reason: d.lines().next().unwrap_or("audit failed").to_string(),
                                },
                                elapsed_ms: batch_start.elapsed().as_millis() as u64,
                            });
                            last_diag = d;
                        }
                    }
                } else if let Some(d) = err {
                    if d.contains("unsolved goals") {
                        unsolved_diags.push((chunk[i].clone(), d.clone()));
                    }
                    last_diag = d.clone();
                }
            }
        } else {
            // Parse-error fallback: re-run candidates individually and collect unsolved goals
            for (i, script) in chunk.iter().enumerate() {
                if let Some(dl) = deadline
                    && Instant::now() >= dl
                {
                    last_diag = "ood escalation budget exceeded".to_string();
                    break;
                }
                let branch_id = chunk_base_id + i + 1;
                match audit_and_prune_unused_simp_args(
                    src,
                    &thm.name,
                    full_name,
                    script,
                    lake_root,
                    prune_unused_simp,
                    strict_lint,
                )? {
                    Ok((cand_src, final_script)) => {
                        let elapsed_ms = batch_start.elapsed().as_millis() as u64;
                        ui.send(UiEvent::BranchUpdated {
                            id: branch_id,
                            status: BranchStatus::Verified {
                                tactic: final_script.join("; "),
                            },
                            elapsed_ms,
                        });
                        return Ok((
                            Some((cand_src, final_script)),
                            String::new(),
                            unsolved_diags,
                        ));
                    }
                    Err(d) => {
                        ui.send(UiEvent::BranchUpdated {
                            id: branch_id,
                            status: BranchStatus::Failed {
                                reason: d.lines().next().unwrap_or("failed").to_string(),
                            },
                            elapsed_ms: batch_start.elapsed().as_millis() as u64,
                        });
                        if d.contains("unsolved goals") {
                            unsolved_diags.push((script.clone(), d.clone()));
                        }
                        last_diag = d;
                    }
                }
            }
        }
    }

    Ok((None, last_diag, unsolved_diags))
}

/// Multi-file worktree candidate specification for concurrent evaluation.
#[derive(Clone, Debug)]
pub struct MultiFileCandidate {
    pub tree: TreeSnapshot,
    pub target_rel_path: PathBuf,
    pub theorem_full_name: String,
    pub script_label: Vec<String>,
    pub simulated_delay_ms: u64,
}

/// Evaluate multi-file `TreeSnapshot` branches concurrently in ephemeral `<temp>/sheaf_wt_<hex12>_*` worktrees
/// using a bounded worker pool (`max_workers`) and early `AtomicBool` cancellation.
pub fn evaluate_multifile_worktree_branches(
    store: &BlobStore,
    base_tree: &TreeSnapshot,
    candidates: &[MultiFileCandidate],
    max_workers: usize,
) -> Result<Option<(TreeSnapshot, Vec<String>)>, ProverError> {
    if candidates.is_empty() {
        return Ok(None);
    }

    let solved = Arc::new(AtomicBool::new(false));
    let worker_cap = max_workers.clamp(1, 8);
    let mut winner: Option<(TreeSnapshot, Vec<String>)> = None;

    for wave in candidates.chunks(worker_cap) {
        if solved.load(Ordering::Acquire) {
            break;
        }
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(wave.len());
            for cand in wave {
                let solved_flag = Arc::clone(&solved);
                handles.push(s.spawn(move || {
                    eval_single_multifile_worktree(store, base_tree, cand, &solved_flag)
                }));
            }
            for h in handles {
                if let Ok(Ok(Some(win))) = h.join()
                    && winner.is_none()
                {
                    winner = Some(win);
                }
            }
        });
    }

    Ok(winner)
}

fn eval_single_multifile_worktree(
    store: &BlobStore,
    base_tree: &TreeSnapshot,
    cand: &MultiFileCandidate,
    solved: &AtomicBool,
) -> Result<Option<(TreeSnapshot, Vec<String>)>, ProverError> {
    if solved.load(Ordering::Acquire) {
        return Ok(None);
    }

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let wt_dir = std::env::temp_dir().join(format!(
        "sheaf_wt_{}_{unique}_{:?}",
        cand.tree.tree_hash.hex12(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&wt_dir)
        .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))?;

    // Materialize dirty files (plus target file if not dirty) into ephemeral worktree
    let dirty = cand.tree.dirty_paths(base_tree);
    for rel_path in dirty {
        if let Some(content) = cand.tree.read_file(store, rel_path) {
            let full_path = wt_dir.join(rel_path);
            if let Some(parent) = full_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&full_path, content.as_ref());
        }
    }

    // Build combined translation unit from materialized files in dependency order
    // (imports at top, non-target helper files next, then target file + #print axioms)
    let mut imports = Vec::new();
    let mut body_src = String::new();
    for (rel_path, hash) in &cand.tree.files {
        if rel_path != &cand.target_rel_path
            && let Some(content) = store.get(hash)
        {
            for line in content.lines() {
                if line.trim_start().starts_with("import ") {
                    if !imports.contains(&line.to_string()) {
                        imports.push(line.to_string());
                    }
                } else {
                    body_src.push_str(line);
                    body_src.push('\n');
                }
            }
            body_src.push('\n');
        }
    }
    if let Some(target_content) = cand.tree.read_file(store, &cand.target_rel_path) {
        for line in target_content.lines() {
            if line.trim_start().starts_with("import ") {
                if !imports.contains(&line.to_string()) {
                    imports.push(line.to_string());
                }
            } else {
                body_src.push_str(line);
                body_src.push('\n');
            }
        }
    }
    let mut combined_src = imports.join("\n");
    if !combined_src.is_empty() {
        combined_src.push('\n');
    }
    combined_src.push_str(&body_src);
    combined_src.push_str(&format!("\n#print axioms {}\n", cand.theorem_full_name));

    let audit_file = wt_dir.join("__multifile_audit.lean");
    if let Err(e) = std::fs::write(&audit_file, &combined_src) {
        let _ = std::fs::remove_dir_all(&wt_dir);
        return Err(ProverError::Repl(lean_repl::ReplError::Io(e)));
    }

    // Optional delay for cancellation testing, checking `solved` every 5ms
    let mut slept = 0u64;
    while slept < cand.simulated_delay_ms {
        if solved.load(Ordering::Acquire) {
            let _ = std::fs::remove_dir_all(&wt_dir);
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(5));
        slept += 5;
    }

    if solved.load(Ordering::Acquire) {
        let _ = std::fs::remove_dir_all(&wt_dir);
        return Ok(None);
    }

    let mut child = match std::process::Command::new("lean")
        .arg("--json")
        .arg(&audit_file)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&wt_dir);
            return Err(ProverError::Repl(lean_repl::ReplError::Io(e)));
        }
    };

    loop {
        if solved.load(Ordering::Acquire) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_dir_all(&wt_dir);
            return Ok(None);
        }
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(e) => {
                let _ = child.kill();
                let _ = std::fs::remove_dir_all(&wt_dir);
                return Err(ProverError::Repl(lean_repl::ReplError::Io(e)));
            }
        }
    }

    let out = child
        .wait_with_output()
        .map_err(|e| ProverError::Repl(lean_repl::ReplError::Io(e)))?;
    let _ = std::fs::remove_dir_all(&wt_dir);

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut has_error = !out.status.success();
    let mut axiom_ok = false;

    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let sev = v.get("severity").and_then(|s| s.as_str()).unwrap_or("");
            let data = v.get("data").and_then(|d| d.as_str()).unwrap_or("");
            if sev == "error" {
                has_error = true;
            }
            if data.contains("depends on axioms:")
                || data.contains("does not depend on any axioms")
            {
                if lean_repl::verify_allowed_axioms(data).is_ok() {
                    axiom_ok = true;
                } else {
                    has_error = true;
                }
            }
        }
    }

    if !has_error && axiom_ok {
        solved.store(true, Ordering::Release);
        Ok(Some((cand.tree.clone(), cand.script_label.clone())))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover_sorry_theorems;

    #[test]
    fn test_streaming_batch_probe_emits_live_branch_updates() {
        let dir = std::env::temp_dir().join("lean_prover_v17_streaming_probe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("StreamProbe.lean");
        let src = "import Init\ntheorem add_comm_stream (a b : Nat) : a + b = b + a := by\n  sorry\n";
        std::fs::write(&f, src).unwrap();
        let thms = discover_sorry_theorems(&f).unwrap();
        let cands = vec![
            vec!["rfl".to_string()],
            vec!["intro h".to_string()],
            vec!["omega".to_string()],
        ];
        let (ui, rx) = UiHandle::test_channel();
        let (solved, _diag, _unsolved) = verify_candidates_batched_with_heartbeats_and_ui(
            src,
            &thms[0],
            "add_comm_stream",
            &cands,
            None,
            200_000,
            &ui,
            0,
        )
        .unwrap();
        drop(ui);
        assert!(solved.is_some());
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(
            events.iter().any(|e| matches!(
                e,
                UiEvent::BranchUpdated {
                    id: 1,
                    status: BranchStatus::Failed { .. },
                    ..
                }
            )),
            "expected live Failed event for branch #1: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                UiEvent::BranchUpdated {
                    id: 3,
                    status: BranchStatus::Verified { .. },
                    ..
                }
            )),
            "expected Verified event for branch #3: {events:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unused_simp_args_pruned_post_verification() {
        let dir = std::env::temp_dir().join("lean_prover_v171_prune_unused_simp");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("UnusedSimp.lean");
        let src = "import Init\ntheorem zero_id_prune (n : Nat) : n + 0 = n := by\n  sorry\n";
        std::fs::write(&f, src).unwrap();
        let thms = discover_sorry_theorems(&f).unwrap();
        let cands = vec![vec!["simp [Nat.add_comm, Nat.mul_comm]".to_string()]];
        let (solved, diag, _) = verify_candidates_batched_with_options_and_ui(
            src,
            &thms[0],
            "zero_id_prune",
            &cands,
            None,
            200_000,
            &UiHandle::noop(),
            0,
            true,
            true,
        )
        .unwrap();
        let (_, script) =
            solved.unwrap_or_else(|| panic!("expected verified proof, got diag: {diag}"));
        assert_eq!(script, vec!["simp".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_per_occurrence_unused_simp_args_in_multi_branch_tactic() {
        let dir = std::env::temp_dir().join("lean_prover_v180_per_occ_prune");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("MultiBranchPrune.lean");
        let src = "\
import Init
theorem branch_simp_prune (n : Nat) (h : n = 0) : n + 0 = 0 ∧ 0 + n = 0 := by
  sorry
";
        std::fs::write(&f, src).unwrap();
        let thms = discover_sorry_theorems(&f).unwrap();
        // First branch does not need Nat.mul_comm; second branch does not need Nat.mul_assoc
        let cands = vec![vec![
            "constructor; simp [h, Nat.mul_comm]; simp [h, Nat.mul_assoc]".to_string(),
        ]];
        let (solved, diag, _) = verify_candidates_batched_with_options_and_ui(
            src,
            &thms[0],
            "branch_simp_prune",
            &cands,
            None,
            200_000,
            &UiHandle::noop(),
            0,
            true,
            true,
        )
        .unwrap();
        let (_, script) =
            solved.unwrap_or_else(|| panic!("expected strict-lint clean proof, got diag: {diag}"));
        assert!(
            !script[0].contains("Nat.mul_comm") && !script[0].contains("Nat.mul_assoc"),
            "expected unused simp args stripped per occurrence: {script:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_strict_lint_rejects_unprunable_warning_without_silent_fallback() {
        let dir = std::env::temp_dir().join("lean_prover_strict_lint_hard_gate_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("StrictReject.lean");
        let src = "\
import Init
theorem strict_reject (unused_n : Nat) : 0 = 0 := by
  sorry
";
        std::fs::write(&f, src).unwrap();
        let thms = discover_sorry_theorems(&f).unwrap();
        let cands = vec![vec!["simp [Nat.add_zero]".to_string()]];
        let (solved, diag, _) = verify_candidates_batched_with_options_and_ui(
            src,
            &thms[0],
            "strict_reject",
            &cands,
            None,
            200_000,
            &UiHandle::noop(),
            0,
            true,
            true,
        )
        .unwrap();
        assert!(
            solved.is_none(),
            "strict_lint must reject unprunable unusedVariables warning, got: {solved:?} ({diag})"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_strict_lint_ignores_sorry_warnings_in_other_theorems() {
        let dir = std::env::temp_dir().join("lean_prover_strict_lint_multitheorem_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("MultiTheorem.lean");
        let src = "\
import Init
theorem thm_one (n : Nat) : n + 0 = n := by
  sorry

theorem thm_two (m : Nat) : m = m := by
  sorry
";
        std::fs::write(&f, src).unwrap();
        let thms = discover_sorry_theorems(&f).unwrap();
        let cands = vec![vec!["simp [Nat.add_zero, Nat.mul_comm]".to_string()]];
        let (solved, diag, _) = verify_candidates_batched_with_options_and_ui(
            src,
            &thms[0],
            "thm_one",
            &cands,
            None,
            200_000,
            &UiHandle::noop(),
            0,
            true,
            true,
        )
        .unwrap();
        let (_, script) = solved.unwrap_or_else(|| {
            panic!("strict_lint must ignore sorry warnings in thm_two, got diag: {diag}")
        });
        assert_eq!(script, vec!["simp [Nat.add_zero]".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_strip_trailing_docstrings_handles_module_docstring() {
        let prefix = concat!(
            "/-- Helper doc -/\n",
            "def route (x : Nat) : Nat := x\n",
            "/-! ### 1. Section Header -/\n",
        );
        let stripped = strip_trailing_docstrings_and_attrs(prefix);
        assert!(
            stripped.contains("def route (x : Nat) : Nat := x"),
            "module docstring /-! ... -/ must not strip preceding def: {stripped}"
        );
        assert!(!stripped.contains("Section Header"));
    }

    #[test]
    fn test_m21_002_probe_preflight_build_and_sanity() {
        let dir = std::env::temp_dir().join("sheaf_m21_002_preflight_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let f_err = dir.join("CompileErr.lean");
        let src_err = "import Init\ndef bad : Nat := true\ntheorem t (n : Nat) : n = n := by\n  sorry\n";
        std::fs::write(&f_err, src_err).unwrap();
        let thms_err = discover_sorry_theorems(&f_err).unwrap();
        let (b_err, _, diag_err) =
            probe_preflight_build_and_sanity(src_err, &thms_err[0], "t", None).unwrap();
        assert_eq!(b_err, crate::BuildStatus::CompileError);
        assert!(diag_err.is_some());

        let f_triv = dir.join("Triv.lean");
        let src_triv = "import Init\ntheorem t_triv (n : Nat) : n + 0 = n := by\n  sorry\n";
        std::fs::write(&f_triv, src_triv).unwrap();
        let thms_triv = discover_sorry_theorems(&f_triv).unwrap();
        let (b_triv, s_triv, tactic_triv) =
            probe_preflight_build_and_sanity(src_triv, &thms_triv[0], "t_triv", None).unwrap();
        assert_eq!(b_triv, crate::BuildStatus::Ok);
        assert_eq!(s_triv, crate::SanityStatus::Trivial);
        assert!(tactic_triv.is_some());

        let f_false = dir.join("FalseSpec.lean");
        let src_false = "import Init\ntheorem t_false : (0 : Nat) = 1 := by\n  sorry\n";
        std::fs::write(&f_false, src_false).unwrap();
        let thms_false = discover_sorry_theorems(&f_false).unwrap();
        let (b_false, s_false, _) =
            probe_preflight_build_and_sanity(src_false, &thms_false[0], "t_false", None).unwrap();
        assert_eq!(b_false, crate::BuildStatus::Ok);
        assert_eq!(s_false, crate::SanityStatus::FalseSpec);

        let f_open = dir.join("Open.lean");
        let src_open = "import Init\ntheorem t_open (xs ys : List Nat) : (xs ++ ys).length = xs.length + ys.length := by\n  sorry\n";
        std::fs::write(&f_open, src_open).unwrap();
        let thms_open = discover_sorry_theorems(&f_open).unwrap();
        let (b_open, s_open, _) =
            probe_preflight_build_and_sanity(src_open, &thms_open[0], "t_open", None).unwrap();
        assert_eq!(b_open, crate::BuildStatus::Ok);
        assert_eq!(s_open, crate::SanityStatus::ValidOpen);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_m23_task3_strict_lint_prunes_compound_simp_all_grind() {
        let src = concat!(
            "import Init\n",
            "def d1 (x : Nat) : Nat := x + 1\n",
            "def d2 (x : Nat) : Nat := x + 2\n",
            "def d3 (x : Nat) : Nat := x + 3\n",
            "def d4 (x : Nat) : Nat := x + 4\n",
            "def d5 (x : Nat) : Nat := x + 5\n",
            "theorem only_d1_needed (a b : Nat) (h1 : a + 1 ≤ b) (h2 : b ≤ a + 1) : d1 a = b := by\n",
            "  sorry\n",
        );
        let script = vec!["simp_all [d1, d2, d3, d4, d5]; grind".to_string()];
        let audited = audit_and_prune_unused_simp_args(
            src,
            "only_d1_needed",
            "only_d1_needed",
            &script,
            None,
            true,
            true,
        )
        .unwrap()
        .expect("compound simp_all; grind must prune unused defs and pass strict_lint");
        let joined = audited.1.join("; ");
        assert!(joined.contains("d1"), "must keep needed d1: {joined}");
        for unused in ["d2", "d3", "d4", "d5"] {
            assert!(!joined.contains(unused), "must prune unused {unused}: {joined}");
        }
    }

    #[test]
    fn test_m24_ood_deadline_aborts_batched_probes_promptly() {
        let src = "import Init\ntheorem slow_fail (n : Nat) : n = n + 1 := by\n  sorry\n";
        let thm = crate::DiscoveredTheorem {
            file: std::path::PathBuf::from("Slow.lean"),
            line: 2,
            namespace: None,
            name: "slow_fail".to_string(),
            signature: "theorem slow_fail (n : Nat) : n = n + 1 := by sorry".to_string(),
            binders: vec![("n".to_string(), "Nat".to_string())],
            local_defs: Vec::new(),
            local_structures: Vec::new(),
            local_inductives: Vec::new(),
            local_theorems: Vec::new(),
            local_struct_laws: Vec::new(),
            has_mathlib: false,
            workspace_deps: Vec::new(),
        };
        let candidates: Vec<Vec<String>> = (0..24)
            .map(|i| vec![format!("have h_{i} : True := trivial; exact absurd rfl (by simp)")])
            .collect();
        let ui = UiHandle::noop();
        let t0 = Instant::now();
        let deadline = Some(t0 + std::time::Duration::from_millis(1));
        let (solved, diag, _) = verify_candidates_batched_with_deadline_and_ui(
            src,
            &thm,
            "slow_fail",
            &candidates,
            None,
            50_000,
            &ui,
            0,
            false,
            false,
            deadline,
        )
        .unwrap();
        assert!(solved.is_none());
        assert!(
            diag.contains("ood escalation budget exceeded"),
            "expected budget exceeded diag, got: {diag}"
        );
        assert!(
            t0.elapsed().as_secs_f64() < 20.0,
            "deadline abort took too long: {:?}",
            t0.elapsed()
        );
    }

    #[test]
    fn test_m25_persistent_lean_server_hot_did_change_speedup() {
        let s1 = "import Init\ndef d1 (x : Nat) : Nat := x\ndef d2 (x : Nat) : Nat := x + 1\ntheorem t_warm (x : Nat) : d1 x = x := by\n  simp [d1, d2]\n#print axioms t_warm\n";
        let _ = eval_via_lean_server(s1, None, None).unwrap();

        let t_hot = Instant::now();
        let s2 = "import Init\ndef d1 (x : Nat) : Nat := x\ndef d2 (x : Nat) : Nat := x + 1\ntheorem t_warm (x : Nat) : d1 x = x := by\n  simp [d1]\n#print axioms t_warm\n";
        let out = eval_via_lean_server(s2, None, None).unwrap();
        let hot_ms = t_hot.elapsed().as_millis();
        let ndjson = match out {
            ServerEvalOutcome::Ndjson(s) => s,
            ServerEvalOutcome::TimedOut => panic!("unexpected timeout on hot didChange"),
        };
        assert!(ndjson.contains("depends on axioms:"), "missing axioms diag: {ndjson}");
        assert!(!ndjson.contains("unusedSimpArgs"), "unexpected unused simp: {ndjson}");
        assert!(hot_ms < 1000, "hot didChange took {hot_ms}ms (expected <1000ms)");
    }

    #[test]
    fn test_m26_plain_goal_extracts_live_subgoal() {
        let src = "import Init\ntheorem swap_goal (a b : Nat) : a + b = b + a := by\n  sorry\n";
        let goals = query_plain_goals_for_script(src, "swap_goal", &["intro".to_string()], None, None)
            .unwrap();
        assert!(!goals.is_empty(), "expected $/lean/plainGoal subgoals");
        assert!(
            goals[0].contains("a + b = b + a"),
            "unexpected plainGoal: {:?}",
            goals
        );
    }
}
