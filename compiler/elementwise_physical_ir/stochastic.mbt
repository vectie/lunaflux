///|
pub(all) enum StochasticFinalStep {
  OrderedSoftmax
  RequireFiniteMass
  MinimalProbabilityPrefix
  SumRetainedMass
  CounterDraw
  OrderedInverseCdf
  PublishSample
} derive(Eq, Debug)

///|
pub extend StochasticFinalStep with Eq::{equal, not_equal}

///|
pub extend StochasticFinalStep with @debug.Debug::{to_repr}

///|
/// Serial F64 probability fold over the previously selected total-order top-k.
/// Prefix selection and the second sum stay distinct: replacing the latter by
/// a subtraction or parallel reduction changes deterministic replay semantics.
pub struct StochasticProgram {
  priv selection : SelectionProgram
  priv maximum_top_k : Int
} derive(Eq, Debug)

///|
pub extend StochasticProgram with Eq::{equal, not_equal}

///|
pub extend StochasticProgram with @debug.Debug::{to_repr}

///|
pub fn StochasticProgram::new(
  values~ : Int,
  owners~ : Int,
  subgroup_lanes~ : Int,
  maximum_top_k~ : Int,
) -> StochasticProgram? {
  guard maximum_top_k > 0 && maximum_top_k <= values else { return None }
  guard SelectionProgram::new(
      values~,
      owners~,
      subgroup_lanes~,
      reduction=OwnerTree,
    )
    is Some(selection) else {
    return None
  }
  Some({ selection, maximum_top_k, })
}

///|
pub fn StochasticProgram::selection(
  self : StochasticProgram,
) -> SelectionProgram {
  self.selection
}

///|
pub fn StochasticProgram::maximum_top_k(self : StochasticProgram) -> Int {
  self.maximum_top_k
}

///|
pub fn StochasticProgram::final_steps(
  _self : StochasticProgram,
) -> ReadOnlyArray[StochasticFinalStep] {
  [
    OrderedSoftmax,
    RequireFiniteMass,
    MinimalProbabilityPrefix,
    SumRetainedMass,
    CounterDraw,
    OrderedInverseCdf,
    PublishSample,
  ]
}
