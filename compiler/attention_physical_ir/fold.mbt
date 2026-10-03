///|
/// Named values of the online fold. Previous and next state are distinct even
/// when a backend realizes them with the same registers after their last use.
pub(all) enum FoldValue {
  QueryOperand
  KeyOperand
  ValueOperand
  VisibilityControl
  ScoreScale
  RawScores
  Scores
  LocalMaximum
  PreviousMaximum
  NextMaximum
  PreviousDenominator
  NextDenominator
  PreviousOutput
  RescaledOutput
  NextOutput
  RescaleFactor
  ProbabilitySum
} derive(Eq, Debug)

///|
pub extend FoldValue with Eq::{equal, not_equal}

///|
pub extend FoldValue with @debug.Debug::{to_repr}

///|
/// These are domain operations, not device instructions. In particular,
/// probability rounding is inside AccumulateProbabilityValue, after computing
/// the unrounded probability sum; moving it earlier changes the denominator.
pub(all) enum FoldOperation {
  ScoreProduct
  PrepareScores
  MergeMaximum
  RescaleOutput
  AccumulateProbabilityValue
  UpdateDenominator
} derive(Eq, Debug)

///|
pub extend FoldOperation with Eq::{equal, not_equal}

///|
pub extend FoldOperation with @debug.Debug::{to_repr}

///|
pub fn FoldOperation::reads(self : FoldOperation) -> ReadOnlyArray[FoldValue] {
  match self {
    ScoreProduct => [QueryOperand, KeyOperand]
    PrepareScores => [RawScores, VisibilityControl, ScoreScale]
    MergeMaximum => [LocalMaximum, PreviousMaximum]
    RescaleOutput => [PreviousOutput, RescaleFactor]
    AccumulateProbabilityValue =>
      [Scores, NextMaximum, RescaledOutput, ValueOperand]
    UpdateDenominator => [PreviousDenominator, RescaleFactor, ProbabilitySum]
  }
}

///|
pub fn FoldOperation::writes(self : FoldOperation) -> ReadOnlyArray[FoldValue] {
  match self {
    ScoreProduct => [RawScores]
    PrepareScores => [Scores, LocalMaximum]
    MergeMaximum => [NextMaximum, RescaleFactor]
    RescaleOutput => [RescaledOutput]
    AccumulateProbabilityValue => [NextOutput, ProbabilitySum]
    UpdateDenominator => [NextDenominator]
  }
}

///|
/// Numerical contract carried independently of memory ownership and schedule.
pub(all) enum AttentionExponentialLaw {
  StrictNaturalExponential
  /// Explicitly permits base-two approximation and F32 subnormal flush. This
  /// is a numerical alternative, not an equality-preserving compiler rewrite.
  ApproximateBaseTwoExponential
} derive(Eq, Debug)

///|
pub extend AttentionExponentialLaw with Eq::{equal, not_equal}

///|
pub extend AttentionExponentialLaw with @debug.Debug::{to_repr}

///|
/// A closed physical state program for register-owned online softmax. Geometry
/// is scalar and construction is constant-space. There is no CUDA, model name,
/// source string, memory allocation or transfer scheduling authority here.
pub struct OnlineFold {
  priv queries : Int
  priv keys : Int
  priv dimensions : Int
  priv exponential : AttentionExponentialLaw
} derive(Eq, Debug)

///|
pub extend OnlineFold with Eq::{equal, not_equal}

///|
pub extend OnlineFold with @debug.Debug::{to_repr}

///|
pub fn OnlineFold::new(
  queries~ : Int,
  keys~ : Int,
  dimensions~ : Int,
) -> OnlineFold? {
  guard queries > 0 &&
    keys > 0 &&
    dimensions > 0 &&
    queries.to_int64() * keys.to_int64() <= 2147483647L &&
    queries.to_int64() * dimensions.to_int64() <= 2147483647L else {
    return None
  }
  Some({ queries, keys, dimensions, exponential: StrictNaturalExponential, })
}

///|
/// Semantic admission supplies the law; scheduling and lowering must retain it.
/// The existing strict program is never mutated or reinterpreted.
pub fn OnlineFold::with_exponential_law(
  self : OnlineFold,
  exponential : AttentionExponentialLaw,
) -> OnlineFold {
  { ..self, exponential, }
}

///|
pub fn OnlineFold::exponential_law(
  self : OnlineFold,
) -> AttentionExponentialLaw {
  self.exponential
}

///|
pub fn OnlineFold::geometry(self : OnlineFold) -> (Int, Int, Int) {
  (self.queries, self.keys, self.dimensions)
}

///|
/// This phase needs only published Q/K. Its results may survive a V transfer
/// wait without recomputing scores. Transfer effects remain schedule-owned.
pub fn OnlineFold::score_phase(
  _self : OnlineFold,
) -> ReadOnlyArray[FoldOperation] {
  [ScoreProduct]
}

///|
/// Ordered state transition after V publication. No pass may reassociate the
/// key fold or merge the unrounded denominator sum with BF16 PV operands.
pub fn OnlineFold::value_phase(
  _self : OnlineFold,
) -> ReadOnlyArray[FoldOperation] {
  [
    PrepareScores,
    MergeMaximum,
    RescaleOutput,
    AccumulateProbabilityValue,
    UpdateDenominator,
  ]
}

///|
pub fn OnlineFold::carried_state(
  _self : OnlineFold,
) -> ReadOnlyArray[(FoldValue, FoldValue)] {
  [
    (NextMaximum, PreviousMaximum),
    (NextDenominator, PreviousDenominator),
    (NextOutput, PreviousOutput),
  ]
}

///|
pub fn OnlineFold::canonical(self : OnlineFold) -> String {
  let strict = "attention-online-fold-v1;queries=\{self.queries};keys=\{self.keys};dimensions=\{self.dimensions};order=ascending-key;probability=bf16-rne;denominator=unrounded-f32;empty-maximum=negative-infinity;empty-denominator=zero;empty-output=zero"
  match self.exponential {
    StrictNaturalExponential => strict
    ApproximateBaseTwoExponential =>
      strict + ";exponential=approx-base2-f32-v1;subnormal=flush-permitted"
  }
}
