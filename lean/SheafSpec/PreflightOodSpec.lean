import Init

namespace SheafSpec

inductive BuildStatus where
  | ok
  | compileError
  deriving DecidableEq, Repr

inductive SanityStatus where
  | validOpen
  | trivial
  | falseSpec
  deriving DecidableEq, Repr

inductive CmdMode where
  | checkOnly
  | prove
  deriving DecidableEq, Repr

inductive OodStatus where
  | inDist
  | oodDomain
  deriving DecidableEq, Repr

inductive RouteOutcome where
  | abortCompileError
  | abortFalseSpec
  | reportOnly
  | directTrivial
  | directExecute
  | escalateVdb
  deriving DecidableEq, Repr

def isProverAction : RouteOutcome → Bool
  | .directTrivial => true
  | .directExecute => true
  | .escalateVdb => true
  | .abortCompileError => false
  | .abortFalseSpec => false
  | .reportOnly => false

/-- Relaxes the System 1 confidence gate when custom domain symbols trigger OOD. -/
def effectiveThreshold (ood : OodStatus) (baseTau oodDrop : Nat) : Nat :=
  match ood with
  | .inDist => baseTau
  | .oodDomain => baseTau - oodDrop

/-- Unified pre-flight and OOD-aware routing policy. -/
def routePipeline
    (mode : CmdMode)
    (build : BuildStatus)
    (sanity : SanityStatus)
    (ood : OodStatus)
    (conf baseTau oodDrop vdbHits : Nat) : RouteOutcome :=
  match build with
  | .compileError => .abortCompileError
  | .ok =>
    match sanity with
    | .falseSpec => .abortFalseSpec
    | .trivial =>
      match mode with
      | .checkOnly => .reportOnly
      | .prove => .directTrivial
    | .validOpen =>
      match mode with
      | .checkOnly => .reportOnly
      | .prove =>
        match ood with
        | .oodDomain =>
          if vdbHits == 0 || conf >= effectiveThreshold .oodDomain baseTau oodDrop then
            .directExecute
          else
            .escalateVdb
        | .inDist =>
          if conf >= effectiveThreshold .inDist baseTau oodDrop then
            .directExecute
          else
            .escalateVdb

-- ### 1. Pre-Flight Safety Invariants

theorem compile_error_aborts
    (mode : CmdMode) (sanity : SanityStatus) (ood : OodStatus)
    (conf baseTau oodDrop vdbHits : Nat) :
    routePipeline mode .compileError sanity ood conf baseTau oodDrop vdbHits = .abortCompileError := by
  simp [routePipeline]

theorem false_spec_aborts
    (mode : CmdMode) (ood : OodStatus)
    (conf baseTau oodDrop vdbHits : Nat) :
    routePipeline mode .ok .falseSpec ood conf baseTau oodDrop vdbHits = .abortFalseSpec := by
  simp [routePipeline]

theorem check_mode_ok_never_proves
    (sanity : SanityStatus) (ood : OodStatus)
    (conf baseTau oodDrop vdbHits : Nat) :
    isProverAction (routePipeline .checkOnly .ok sanity ood conf baseTau oodDrop vdbHits) = false := by
  induction sanity generalizing ood conf baseTau oodDrop vdbHits <;> simp_all [isProverAction, routePipeline]

theorem check_mode_never_proves
    (build : BuildStatus) (sanity : SanityStatus) (ood : OodStatus)
    (conf baseTau oodDrop vdbHits : Nat) :
    isProverAction (routePipeline .checkOnly build sanity ood conf baseTau oodDrop vdbHits) = false := by
  cases build <;> simp [compile_error_aborts, check_mode_ok_never_proves] <;> simp [isProverAction]

theorem trivial_bypasses_search
    (ood : OodStatus) (conf baseTau oodDrop vdbHits : Nat) :
    routePipeline .prove .ok .trivial ood conf baseTau oodDrop vdbHits = .directTrivial := by
  simp [routePipeline]

-- ### 2. OOD Routing Invariants

theorem ood_relaxes_threshold (baseTau oodDrop : Nat) :
    effectiveThreshold .oodDomain baseTau oodDrop ≤ effectiveThreshold .inDist baseTau oodDrop := by
  dsimp [effectiveThreshold] at *; omega

theorem ood_zero_vdb_hits_avoids_loop
    (conf baseTau oodDrop : Nat) :
    routePipeline .prove .ok .validOpen .oodDomain conf baseTau oodDrop 0 = .directExecute := by
  simp [effectiveThreshold, routePipeline]

theorem in_dist_low_conf_escalates
    (conf baseTau oodDrop vdbHits : Nat) (h : conf < baseTau) :
    routePipeline .prove .ok .validOpen .inDist conf baseTau oodDrop vdbHits = .escalateVdb := by
  simp [effectiveThreshold, routePipeline] <;> all_goals (repeat (first | rfl | (split <;> try simp_all [isProverAction, effectiveThreshold, routePipeline, compile_error_aborts, false_spec_aborts, check_mode_never_proves, trivial_bypasses_search, ood_relaxes_threshold, ood_zero_vdb_hits_avoids_loop]) | omega))

-- ### 3. Cross-File OOD, Exit Codes & OOD Timeout Guard

def classifyOod (hasLocalCustom hasWorkspaceDeps : Bool) : OodStatus :=
  if hasLocalCustom || hasWorkspaceDeps then .oodDomain else .inDist

theorem workspace_deps_triggers_ood (hasLocal : Bool) :
    classifyOod hasLocal true = .oodDomain := by
  simp_all [classifyOod]

def exitCode (outcome : RouteOutcome) (proved : Bool) : Nat :=
  match outcome with
  | .abortCompileError => 2
  | .abortFalseSpec => 3
  | .reportOnly => 0
  | .directTrivial => 0
  | .directExecute => if proved then 0 else 1
  | .escalateVdb => if proved then 0 else 1

theorem compile_error_exit_2 (proved : Bool) :
    exitCode .abortCompileError proved = 2 := by
  first | rfl | decide | trivial

theorem false_spec_exit_3 (proved : Bool) :
    exitCode .abortFalseSpec proved = 3 := by
  first | rfl | decide | trivial

def shouldAbortOodSearch (ood : OodStatus) (elapsedMs timeoutMs : Nat) : Bool :=
  match ood with
  | .oodDomain => decide (elapsedMs >= timeoutMs)
  | .inDist => false

theorem ood_budget_exceeded_stops (elapsedMs timeoutMs : Nat) (h : elapsedMs ≥ timeoutMs) :
    shouldAbortOodSearch .oodDomain elapsedMs timeoutMs = true := by
  induction elapsedMs <;> simp_all [shouldAbortOodSearch]

end SheafSpec
