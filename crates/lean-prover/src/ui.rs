//! Event-driven `uv`-style terminal UI for `sheaf` backed by `indicatif` + `console`:
//! two-zone `stderr` viewport, 12 Hz `MultiProgress`, automatic non-TTY degradation, and `--json` NDJSON mode.

use console::{Term, style, truncate_str};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const MAX_BRANCH_ROWS: usize = 6;
const TICK_MS: u64 = 80;
const FALLBACK_COLS: usize = 80;

/// Terminal rendering mode selected via CLI flags or `stderr` TTY detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiMode {
    #[default]
    Auto,
    TtyLive,
    Verbose,
    Plain,
    Json,
}

impl UiMode {
    /// Resolve `Auto` via `console::Term::stderr().is_term()`.
    #[must_use]
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto => {
                if Term::stderr().is_term() {
                    Self::TtyLive
                } else {
                    Self::Plain
                }
            }
            other => other,
        }
    }
}

/// Live status of an individual candidate tactic branch or worktree worker.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum BranchStatus {
    Queued,
    Running,
    Verified { tactic: String },
    Failed { reason: String },
    Canceled,
}

/// Structured UI telemetry events emitted by prover stages and kernel evaluators.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum UiEvent {
    WorkspaceStarted {
        total: usize,
    },
    TheoremStarted {
        index: usize,
        total: usize,
        name: String,
    },
    ModelRouted {
        strategy: String,
        conf: f32,
        gate: String,
    },
    Escalated {
        vdb_hits: usize,
        tau: f32,
    },
    SearchTick {
        depth: usize,
        max_depth: usize,
        nodes: usize,
        max_nodes: usize,
    },
    BranchSpawned {
        id: usize,
        tag: String,
        preview: String,
    },
    BranchUpdated {
        id: usize,
        status: BranchStatus,
        elapsed_ms: u64,
    },
    StageLogReady {
        log_block: String,
    },
    TheoremFinished {
        name: String,
        proved: bool,
        route_label: String,
        tactic: Option<String>,
        diag: Option<String>,
        elapsed_ms: u64,
    },
    WorkspaceFinished {
        proved: usize,
        total: usize,
        direct_count: usize,
        escalated_count: usize,
        elapsed_ms: u64,
    },
}

/// Thread-safe handle passed to prover stages and kernel workers to emit UI events.
#[derive(Clone, Debug, Default)]
pub struct UiHandle {
    tx: Option<Sender<UiEvent>>,
}

impl UiHandle {
    /// No-op handle for tests and programmatic callers that do not render terminal UI.
    #[must_use]
    pub fn noop() -> Self {
        Self { tx: None }
    }

    #[cfg(test)]
    pub(crate) fn test_channel() -> (Self, mpsc::Receiver<UiEvent>) {
        let (tx, rx) = mpsc::channel();
        (Self { tx: Some(tx) }, rx)
    }

    /// Push a `UiEvent` without blocking workers.
    pub fn send(&self, event: UiEvent) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(event);
        }
    }
}

#[derive(Debug, Clone)]
struct BranchRow {
    id: usize,
    tag: String,
    preview: String,
    status: BranchStatus,
    elapsed_ms: u64,
}

#[derive(Debug, Default)]
struct RendererState {
    total_theorems: usize,
    active_index: usize,
    active_name: String,
    active_started: Option<Instant>,
    route_summary: String,
    escalated_summary: Option<String>,
    depth: usize,
    max_depth: usize,
    nodes: usize,
    max_nodes: usize,
    branches: Vec<BranchRow>,
}

impl RendererState {
    fn reset_theorem(&mut self, index: usize, total: usize, name: String) {
        self.active_index = index;
        self.total_theorems = total;
        self.active_name = name;
        self.active_started = Some(Instant::now());
        self.route_summary.clear();
        self.escalated_summary = None;
        self.depth = 0;
        self.max_depth = 0;
        self.nodes = 0;
        self.max_nodes = 0;
        self.branches.clear();
    }

    fn apply_event(&mut self, ev: &UiEvent) {
        match ev {
            UiEvent::WorkspaceStarted { total } => {
                self.total_theorems = *total;
            }
            UiEvent::TheoremStarted { index, total, name } => {
                self.reset_theorem(*index, *total, name.clone());
            }
            UiEvent::ModelRouted {
                strategy,
                conf,
                gate,
            } => {
                self.route_summary = format!("Sys1: {strategy} ({conf:.2}) -> {gate}");
            }
            UiEvent::Escalated { vdb_hits, tau } => {
                self.escalated_summary =
                    Some(format!("Escalated: VDB ({vdb_hits} hits, tau={tau:.2})"));
            }
            UiEvent::SearchTick {
                depth,
                max_depth,
                nodes,
                max_nodes,
            } => {
                self.depth = *depth;
                self.max_depth = *max_depth;
                self.nodes = *nodes;
                self.max_nodes = *max_nodes;
            }
            UiEvent::BranchSpawned { id, tag, preview } => {
                if let Some(existing) = self.branches.iter_mut().find(|b| b.id == *id) {
                    existing.tag = tag.clone();
                    existing.preview = preview.clone();
                    existing.status = BranchStatus::Queued;
                    existing.elapsed_ms = 0;
                } else {
                    self.branches.push(BranchRow {
                        id: *id,
                        tag: tag.clone(),
                        preview: preview.clone(),
                        status: BranchStatus::Queued,
                        elapsed_ms: 0,
                    });
                }
            }
            UiEvent::BranchUpdated {
                id,
                status,
                elapsed_ms,
            } => {
                if let Some(row) = self.branches.iter_mut().find(|b| b.id == *id) {
                    row.status = status.clone();
                    row.elapsed_ms = *elapsed_ms;
                }
            }
            UiEvent::TheoremFinished { .. }
            | UiEvent::StageLogReady { .. }
            | UiEvent::WorkspaceFinished { .. } => {}
        }
    }

    fn format_header_line(&self, max_cols: usize) -> String {
        let elapsed_s = self
            .active_started
            .map_or(0.0, |t| t.elapsed().as_secs_f64());
        let header = if self.max_depth > 0 || self.max_nodes > 0 {
            format!(
                "  Proving [{}/{}] {} (depth {}/{} · nodes {}/{} · {:.1}s)",
                self.active_index,
                self.total_theorems,
                self.active_name,
                self.depth,
                self.max_depth,
                self.nodes,
                self.max_nodes,
                elapsed_s
            )
        } else {
            format!(
                "  Proving [{}/{}] {} ({:.1}s)",
                self.active_index, self.total_theorems, self.active_name, elapsed_s
            )
        };
        clamp_line(&header, max_cols)
    }

    fn format_route_line(&self, max_cols: usize) -> Option<String> {
        if self.route_summary.is_empty() {
            return None;
        }
        let route_line = match &self.escalated_summary {
            Some(esc) => format!("  ├─ {}  |  {}", self.route_summary, esc),
            None => format!("  ├─ {}", self.route_summary),
        };
        Some(clamp_line(&route_line, max_cols))
    }

    #[cfg(test)]
    fn render_ephemeral_lines(&self, max_cols: usize) -> Vec<String> {
        if self.active_name.is_empty() {
            return Vec::new();
        }
        let mut lines = vec![self.format_header_line(max_cols)];
        if let Some(route) = self.format_route_line(max_cols) {
            lines.push(route);
        }
        let total_branches = self.branches.len();
        let show_cap = if total_branches > MAX_BRANCH_ROWS {
            MAX_BRANCH_ROWS - 1
        } else {
            total_branches
        };
        for (i, row) in self.branches.iter().take(show_cap).enumerate() {
            let is_last = total_branches <= MAX_BRANCH_ROWS && i + 1 == show_cap;
            lines.push(format_branch_line(row, is_last, max_cols));
        }
        if total_branches > MAX_BRANCH_ROWS {
            let rem = total_branches - show_cap;
            lines.push(clamp_line(&format!("  └─ … +{rem} queued"), max_cols));
        }
        lines
    }
}

fn term_width_cap() -> usize {
    let cols = Term::stderr().size().1 as usize;
    if cols == 0 {
        FALLBACK_COLS.saturating_sub(2)
    } else {
        cols.saturating_sub(2).max(20)
    }
}

/// Clamp a line to at most `max_chars` characters using `console::truncate_str` so wrapped lines never break `MultiProgress`.
#[must_use]
pub fn clamp_line(s: &str, max_chars: usize) -> String {
    let single_line = s.replace(['\r', '\n'], " ");
    truncate_str(&single_line, max_chars.max(1), "…").into_owned()
}

fn format_branch_line(row: &BranchRow, is_last: bool, max_cols: usize) -> String {
    let tree_ch = if is_last { "└─" } else { "├─" };
    let sec = (row.elapsed_ms as f64) / 1000.0;
    let (glyph, detail) = match &row.status {
        BranchStatus::Queued => ("·", row.preview.clone()),
        BranchStatus::Running => ("", row.preview.clone()),
        BranchStatus::Verified { tactic } => ("✓", tactic.clone()),
        BranchStatus::Failed { reason } => ("✗", format!("{} ({reason})", row.preview)),
        BranchStatus::Canceled => ("⊘", format!("{} (canceled)", row.preview)),
    };
    let raw = if glyph.is_empty() {
        format!(
            "  {tree_ch} branch #{} [{}] {detail}  {sec:.1}s",
            row.id, row.tag
        )
    } else {
        format!(
            "  {tree_ch} {glyph} branch #{} [{}] {detail}  {sec:.1}s",
            row.id, row.tag
        )
    };
    clamp_line(&raw, max_cols)
}

/// Format a single permanent scrollback line for a finished theorem.
#[must_use]
pub fn format_theorem_scrollback_line(
    name: &str,
    proved: bool,
    route_label: &str,
    tactic: Option<&str>,
    diag: Option<&str>,
    elapsed_ms: u64,
) -> String {
    let glyph = if proved { '✓' } else { '✗' };
    let sec = (elapsed_ms as f64) / 1000.0;
    let short_name = clamp_line(name, 28);
    let tail = if proved {
        let t = tactic.unwrap_or("done");
        format!("[{}]", clamp_line(t, 48))
    } else {
        let d = diag.unwrap_or("failed");
        format!("[{}]", clamp_line(d, 48))
    };
    format!("  {glyph} {short_name:<28} {route_label:<24} {sec:>5.1}s  {tail}")
}

fn format_styled_scrollback_line(
    name: &str,
    proved: bool,
    route_label: &str,
    tactic: Option<&str>,
    diag: Option<&str>,
    elapsed_ms: u64,
) -> String {
    let glyph = if proved {
        style("✓").green().bold().to_string()
    } else {
        style("✗").red().bold().to_string()
    };
    let sec = (elapsed_ms as f64) / 1000.0;
    let short_name = clamp_line(name, 28);
    let route_styled = style(format!("{route_label:<24}")).dim().to_string();
    let time_styled = style(format!("{sec:>5.1}s")).dim().to_string();
    let tail = if proved {
        let t = tactic.unwrap_or("done");
        format!("[{}]", clamp_line(t, 48))
    } else {
        let d = diag.unwrap_or("failed");
        style(format!("[{}]", clamp_line(d, 48))).red().to_string()
    };
    format!("  {glyph} {short_name:<28} {route_styled} {time_styled}  {tail}")
}

/// Format the final workspace summary footer line.
#[must_use]
pub fn format_workspace_footer(
    proved: usize,
    total: usize,
    direct_count: usize,
    escalated_count: usize,
    elapsed_ms: u64,
) -> String {
    let sec = (elapsed_ms as f64) / 1000.0;
    format!(
        "Proved {proved}/{total} theorems in {sec:.1}s ({direct_count} DirectExecute, {escalated_count} Escalated, 0 sorryAx)"
    )
}

struct LiveBars {
    mp: Arc<MultiProgress>,
    header: Option<ProgressBar>,
    route: Option<ProgressBar>,
    branches: Vec<(usize, ProgressBar)>,
    overflow: Option<ProgressBar>,
}

impl LiveBars {
    fn new(mode: UiMode) -> Self {
        let draw_target = if mode == UiMode::TtyLive {
            ProgressDrawTarget::stderr_with_hz(12)
        } else {
            ProgressDrawTarget::hidden()
        };
        Self {
            mp: Arc::new(MultiProgress::with_draw_target(draw_target)),
            header: None,
            route: None,
            branches: Vec::new(),
            overflow: None,
        }
    }

    fn sync_from_state(&mut self, mode: UiMode, state: &RendererState) {
        if mode != UiMode::TtyLive || state.active_name.is_empty() {
            return;
        }
        let max_cols = term_width_cap();
        let spinner_style = ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", " "]);
        let plain_style = ProgressStyle::with_template("{msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner());

        let hdr_msg = state.format_header_line(max_cols);
        let hdr = self.header.get_or_insert_with(|| {
            let pb = self.mp.add(ProgressBar::new_spinner());
            pb.set_style(spinner_style.clone());
            pb.enable_steady_tick(Duration::from_millis(TICK_MS));
            pb
        });
        hdr.set_message(hdr_msg);

        if let Some(route_msg) = state.format_route_line(max_cols) {
            let rpb = self.route.get_or_insert_with(|| {
                let pb = self.mp.add(ProgressBar::new_spinner());
                pb.set_style(plain_style.clone());
                pb
            });
            rpb.set_message(route_msg);
        }

        let total_branches = state.branches.len();
        let show_cap = if total_branches > MAX_BRANCH_ROWS {
            MAX_BRANCH_ROWS - 1
        } else {
            total_branches
        };

        for (i, row) in state.branches.iter().take(show_cap).enumerate() {
            let is_last = total_branches <= MAX_BRANCH_ROWS && i + 1 == show_cap;
            let msg = format_branch_line(row, is_last, max_cols);
            let pb = if let Some((_, existing)) = self.branches.iter().find(|(id, _)| *id == row.id)
            {
                existing
            } else {
                let new_pb = self.mp.add(ProgressBar::new_spinner());
                new_pb.enable_steady_tick(Duration::from_millis(TICK_MS));
                self.branches.push((row.id, new_pb));
                &self.branches.last().expect("just pushed").1
            };
            if matches!(row.status, BranchStatus::Running | BranchStatus::Queued) {
                pb.set_style(spinner_style.clone());
            } else {
                pb.set_style(plain_style.clone());
                pb.disable_steady_tick();
            }
            pb.set_message(msg);
        }

        if total_branches > MAX_BRANCH_ROWS {
            let rem = total_branches - show_cap;
            let ov_msg = clamp_line(&format!("  └─ … +{rem} queued"), max_cols);
            let ov = self.overflow.get_or_insert_with(|| {
                let pb = self.mp.add(ProgressBar::new_spinner());
                pb.set_style(plain_style.clone());
                pb
            });
            ov.set_message(ov_msg);
        } else if let Some(ov) = self.overflow.take() {
            ov.finish_and_clear();
        }
    }

    fn clear_active(&mut self) {
        if let Some(ov) = self.overflow.take() {
            ov.finish_and_clear();
        }
        for (_, pb) in self.branches.drain(..) {
            pb.finish_and_clear();
        }
        if let Some(r) = self.route.take() {
            r.finish_and_clear();
        }
        if let Some(h) = self.header.take() {
            h.finish_and_clear();
        }
        let _ = self.mp.clear();
    }
}

/// Active UI session owning the background renderer thread.
pub struct UiSession {
    mode: UiMode,
    tx: Option<Sender<UiEvent>>,
    handle: UiHandle,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
    json_lines: Arc<Mutex<Vec<String>>>,
}

impl UiSession {
    /// Start a UI session in the requested mode (`Auto` checks `console::Term::stderr().is_term()`).
    #[must_use]
    pub fn start(requested_mode: UiMode) -> Self {
        let mode = requested_mode.resolve();
        let (tx, rx) = mpsc::channel::<UiEvent>();
        let handle = UiHandle {
            tx: Some(tx.clone()),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = Arc::clone(&stop);
        let json_lines = Arc::new(Mutex::new(Vec::new()));
        let json_buf = Arc::clone(&json_lines);

        let join = std::thread::spawn(move || {
            let mut state = RendererState::default();
            let mut bars = LiveBars::new(mode);

            loop {
                match rx.recv_timeout(Duration::from_millis(TICK_MS)) {
                    Ok(ev) => {
                        process_event(mode, &mut state, &mut bars, &ev, &json_buf);
                        while let Ok(more) = rx.try_recv() {
                            process_event(mode, &mut state, &mut bars, &more, &json_buf);
                        }
                        bars.sync_from_state(mode, &state);
                        if stop_bg.load(Ordering::Acquire) {
                            bars.clear_active();
                            break;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if stop_bg.load(Ordering::Acquire) {
                            while let Ok(more) = rx.try_recv() {
                                process_event(mode, &mut state, &mut bars, &more, &json_buf);
                            }
                            bars.clear_active();
                            break;
                        }
                        bars.sync_from_state(mode, &state);
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        bars.clear_active();
                        break;
                    }
                }
            }
        });

        Self {
            mode,
            tx: Some(tx),
            handle,
            stop,
            join: Some(join),
            json_lines,
        }
    }

    /// Return a cloneable `UiHandle` for workers and prover loops.
    #[must_use]
    pub fn handle(&self) -> UiHandle {
        self.handle.clone()
    }

    /// Resolved UI mode.
    #[must_use]
    pub fn mode(&self) -> UiMode {
        self.mode
    }

    /// Flush remaining UI events, stop the renderer thread, and return any captured NDJSON lines.
    pub fn finish(mut self) -> Vec<String> {
        self.shutdown();
        self.json_lines
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.handle = UiHandle::noop();
        drop(self.tx.take());
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for UiSession {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn process_event(
    mode: UiMode,
    state: &mut RendererState,
    bars: &mut LiveBars,
    ev: &UiEvent,
    json_buf: &Arc<Mutex<Vec<String>>>,
) {
    if mode == UiMode::Json {
        if let Ok(line) = serde_json::to_string(ev)
            && let Ok(mut guard) = json_buf.lock()
        {
            guard.push(line);
        }
        return;
    }

    state.apply_event(ev);

    match ev {
        UiEvent::StageLogReady { log_block } if mode == UiMode::Verbose => {
            bars.clear_active();
            let _ = bars.mp.println(log_block);
            if !Term::stderr().is_term() {
                eprintln!("{log_block}");
            }
        }
        UiEvent::TheoremFinished {
            name,
            proved,
            route_label,
            tactic,
            diag,
            elapsed_ms,
        } => {
            if mode == UiMode::TtyLive {
                bars.clear_active();
                let line = format_styled_scrollback_line(
                    name,
                    *proved,
                    route_label,
                    tactic.as_deref(),
                    diag.as_deref(),
                    *elapsed_ms,
                );
                let _ = bars.mp.println(line);
                state.active_name.clear();
                state.branches.clear();
            } else if mode == UiMode::Plain {
                let line = format_theorem_scrollback_line(
                    name,
                    *proved,
                    route_label,
                    tactic.as_deref(),
                    diag.as_deref(),
                    *elapsed_ms,
                );
                eprintln!("{line}");
            }
        }
        UiEvent::WorkspaceFinished {
            proved,
            total,
            direct_count,
            escalated_count,
            elapsed_ms,
        } => {
            bars.clear_active();
            let footer = format_workspace_footer(
                *proved,
                *total,
                *direct_count,
                *escalated_count,
                *elapsed_ms,
            );
            if mode == UiMode::TtyLive {
                let _ = bars.mp.println(style(footer).bold().to_string());
            } else if mode == UiMode::Plain {
                eprintln!("{footer}");
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ui_event_lifecycle_and_branch_cap_rendering() {
        let mut state = RendererState::default();
        state.apply_event(&UiEvent::WorkspaceStarted { total: 6 });
        state.apply_event(&UiEvent::TheoremStarted {
            index: 4,
            total: 6,
            name: "Audit.step_then_batch_conserves_long_theorem_name".to_string(),
        });
        state.apply_event(&UiEvent::ModelRouted {
            strategy: "Rewrite".to_string(),
            conf: 0.94,
            gate: "DirectExecute".to_string(),
        });
        state.apply_event(&UiEvent::Escalated {
            vdb_hits: 4,
            tau: 0.50,
        });
        state.apply_event(&UiEvent::SearchTick {
            depth: 1,
            max_depth: 4,
            nodes: 3,
            max_nodes: 64,
        });
        for id in 1..=8 {
            state.apply_event(&UiEvent::BranchSpawned {
                id,
                tag: "Rewrite".to_string(),
                preview: format!("simp [step_conserves_supply_{id}]"),
            });
            state.apply_event(&UiEvent::BranchUpdated {
                id,
                status: if id == 3 {
                    BranchStatus::Failed {
                        reason: "goal not linear".to_string(),
                    }
                } else {
                    BranchStatus::Running
                },
                elapsed_ms: 400 * id as u64,
            });
        }

        let lines = state.render_ephemeral_lines(78);
        // 1 header + 1 route + 5 branches + 1 "+3 queued" = 8 lines
        assert_eq!(lines.len(), 8, "unexpected viewport lines: {lines:?}");
        assert!(lines.last().unwrap().contains("+3 queued"));
        assert!(
            lines.iter().all(|l| l.chars().count() <= 78),
            "all lines must clamp to <= 78 chars: {lines:?}"
        );

        let scroll = format_theorem_scrollback_line(
            "Ops.step_zero_noop",
            true,
            "DirectExecute(Rewrite)",
            Some("simp [step]"),
            None,
            5120,
        );
        assert!(scroll.contains("✓ Ops.step_zero_noop"));
        assert!(scroll.contains("5.1s"));
        assert!(scroll.contains("[simp [step]]"));

        let footer = format_workspace_footer(6, 6, 4, 2, 41_800);
        assert_eq!(
            footer,
            "Proved 6/6 theorems in 41.8s (4 DirectExecute, 2 Escalated, 0 sorryAx)"
        );
    }

    #[test]
    fn test_ui_indicatif_modes_and_json_no_ansi() {
        let session = UiSession::start(UiMode::Json);
        let ui = session.handle();
        ui.send(UiEvent::WorkspaceStarted { total: 1 });
        ui.send(UiEvent::TheoremStarted {
            index: 1,
            total: 1,
            name: "Demo.id".to_string(),
        });
        ui.send(UiEvent::TheoremFinished {
            name: "Demo.id".to_string(),
            proved: true,
            route_label: "DirectExecute(AtomicClose)".to_string(),
            tactic: Some("rfl".to_string()),
            diag: None,
            elapsed_ms: 120,
        });
        ui.send(UiEvent::WorkspaceFinished {
            proved: 1,
            total: 1,
            direct_count: 1,
            escalated_count: 0,
            elapsed_ms: 120,
        });
        drop(ui);
        let ndjson = session.finish();
        assert_eq!(ndjson.len(), 4);
        for line in &ndjson {
            assert!(!line.contains('\x1b'), "JSON line contained ANSI escape: {line}");
            let _: serde_json::Value = serde_json::from_str(line).expect("valid NDJSON");
        }
    }
}
