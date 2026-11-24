import Init

namespace SheafSpec

inductive GateAction where
  | directExecute
  | fallbackToLeJepa
  deriving DecidableEq, Repr

def routeGate (conf tau : Nat) : GateAction :=
  if conf >= tau then .directExecute else .fallbackToLeJepa

theorem gate_zero_threshold (conf : Nat) :
    routeGate conf 0 = .directExecute := by
  simp [routeGate]

theorem gate_escalation_fallback (conf tau : Nat) (h : conf < tau) :
    routeGate conf tau = .fallbackToLeJepa := by
  simp [routeGate] <;> all_goals (repeat (first | rfl | (split <;> try simp_all [routeGate, gate_zero_threshold]) | omega))

end SheafSpec
