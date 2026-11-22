import Init

namespace SheafSpec

def mapGet (m : List (Nat × Nat)) (k : Nat) : Option Nat :=
  match m with
  | [] => none
  | (k', v) :: rest => if k' == k then some v else mapGet rest k

def mapPut (m : List (Nat × Nat)) (k v : Nat) : List (Nat × Nat) :=
  (k, v) :: m

theorem vfs_read_after_write (m : List (Nat × Nat)) (k v : Nat) :
    mapGet (mapPut m k v) k = some v := by
  sorry

theorem vfs_distinct_key_isolated (m : List (Nat × Nat)) (k1 k2 v : Nat) (hne : k1 ≠ k2) :
    mapGet (mapPut m k1 v) k2 = mapGet m k2 := by
  sorry

theorem vfs_overwrite_shadows (m : List (Nat × Nat)) (k v1 v2 : Nat) :
    mapGet (mapPut (mapPut m k v1) k v2) k = some v2 := by
  sorry

end SheafSpec
