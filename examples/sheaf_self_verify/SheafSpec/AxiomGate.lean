import Init

namespace SheafSpec

inductive LeanAxiom where
  | propext
  | classicalChoice
  | quotSound
  | sorryAx
  | custom
  deriving DecidableEq, Repr

def isAllowedAxiom : LeanAxiom → Bool
  | .propext => true
  | .classicalChoice => true
  | .quotSound => true
  | .sorryAx => false
  | .custom => false

def allowlistOk : List LeanAxiom → Bool
  | [] => true
  | ax :: rest => isAllowedAxiom ax && allowlistOk rest

theorem allowlist_nil : allowlistOk [] = true := by
  sorry

theorem verify_rejects_sorry (rest : List LeanAxiom) :
    allowlistOk (.sorryAx :: rest) = false := by
  sorry

theorem allowlist_append (xs ys : List LeanAxiom) :
    allowlistOk (xs ++ ys) = (allowlistOk xs && allowlistOk ys) := by
  sorry

end SheafSpec
