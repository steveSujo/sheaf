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
  simp [allowlistOk]

theorem verify_rejects_sorry (rest : List LeanAxiom) :
    allowlistOk (.sorryAx :: rest) = false := by
  induction rest <;> simp_all [isAllowedAxiom, allowlistOk]

theorem allowlist_append (xs ys : List LeanAxiom) :
    allowlistOk (xs ++ ys) = (allowlistOk xs && allowlistOk ys) := by
  induction xs generalizing ys with | nil => simp_all [allowlistOk] | cons hd tl ih => cases hd <;> simp_all [isAllowedAxiom, allowlistOk, verify_rejects_sorry] <;> cases ys with | nil => simp_all [isAllowedAxiom, allowlistOk, allowlist_nil, verify_rejects_sorry, List.append_assoc] | cons _ tl2 => cases tl2 <;> simp_all [isAllowedAxiom, allowlistOk, allowlist_nil, verify_rejects_sorry, List.append_assoc]

end SheafSpec
