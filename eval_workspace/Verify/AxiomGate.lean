import Init

namespace Verify

/-- Allowed standard Lean 4 axioms for verified solutions. -/
def allowedAxioms : List String :=
  ["propext", "Classical.choice", "Quot.sound"]

/-- Banned unsound or cheat axioms. -/
def bannedAxioms : List String :=
  ["sorryAx", "Lean.ofReduceBool", "Lean.trustCompiler"]

end Verify
