///|
/// A contiguous logical-axis partition. No device topology or instruction
/// geometry participates in this immutable plan.
pub struct TensorAxisPartition {
  priv start : Int
  priv width : Int
}

///|
pub fn TensorAxisPartition::new(
  full_width : Int,
  rank~ : Int,
  world_size~ : Int,
) -> Self raise PrecisionPlanError {
  guard full_width > 0 &&
    world_size > 1 &&
    rank >= 0 &&
    rank < world_size &&
    full_width % world_size == 0 else {
    raise IllegalPlacement
  }
  let width = full_width / world_size
  { start: rank * width, width, }
}

///|
pub fn TensorAxisPartition::range(self : Self) -> (Int, Int) {
  (self.start, self.width)
}

///|
/// Publication and inter-rank addition are ordered numerical operations.
/// Round-before-reduce is a different program, not an algebraic equivalence:
/// each rank rounds locally, then the collective publishes a BF16 sum.
/// Transport remains an explicit effect outside the local projection.
pub(all) enum ProjectionPublication {
  LocalBFloat16
  ReduceFloat32BeforeRound
  ReduceBFloat16AfterRound
} derive(Eq, Debug)

///|
pub extend ProjectionPublication with Eq::{equal, not_equal}

///|
pub extend ProjectionPublication with @debug.Debug::{to_repr}
