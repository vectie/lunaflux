///|
/// A finite two-level maximum tree. Only finite, nonnegative operands enter
/// the tree; non-finite rejection is a separate effect. Max on this domain is
/// associative and exact, unlike floating-point sum. Leaf and group sizes are
/// supplied by device planning rather than encoded as arithmetic semantics.
pub struct MaximumReductionTree {
  priv leaves : Int
  priv group : Int
} derive(Eq, Debug)

///|
pub extend MaximumReductionTree with Eq::{equal, not_equal}

///|
pub extend MaximumReductionTree with @debug.Debug::{to_repr}

///|
pub fn MaximumReductionTree::new(leaves~ : Int, group~ : Int) -> Self? {
  guard leaves > 0 && leaves <= 1024 && group > 0 &&
    group <= leaves && (group & (group - 1)) == 0 &&
    leaves % group == 0 && leaves / group <= group else { return None }
  Some({ leaves, group, })
}

///|
pub fn MaximumReductionTree::geometry(self : Self) -> (Int, Int, Int) {
  (self.leaves, self.group, self.leaves / self.group)
}
