///|
/// An immutable map/reduce partition of the vocabulary. Each segment emits
/// (greatest finite value, lowest tied token, lowest non-finite token). Combining
/// these summaries is associative: no floating-point sum is reassociated.
pub struct GreedySegmentPlan {
  priv vocabulary_size : Int32
  priv maximum_rows : Int32
  priv segment_width : Int32
  priv segment_count : Int32
  priv workspace_bytes : Int64
} derive(Debug, Eq)

///|
pub extend GreedySegmentPlan with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend GreedySegmentPlan with Eq::{not_equal, equal}

///|
pub fn plan_greedy_segments(
  vocabulary_size~ : Int32,
  maximum_rows~ : Int32,
  segment_width~ : Int32,
) -> GreedySegmentPlan raise SamplingStrategyError {
  if vocabulary_size <= 0 || maximum_rows <= 0 || segment_width <= 0 {
    raise InvalidSamplingStrategy(Shape)
  }
  let segment_count = (vocabulary_size - 1) / segment_width + 1
  let cells = maximum_rows.to_int64() * segment_count.to_int64()
  if cells > 9223372036854775807L / 16L {
    raise InvalidSamplingStrategy(Workspace)
  }
  {
    vocabulary_size,
    maximum_rows,
    segment_width,
    segment_count,
    workspace_bytes: cells * 16L,
  }
}

///|
pub fn GreedySegmentPlan::shape(self : GreedySegmentPlan) -> (Int32, Int32) {
  (self.maximum_rows, self.vocabulary_size)
}

///|
pub fn GreedySegmentPlan::partition(self : GreedySegmentPlan) -> (Int32, Int32) {
  (self.segment_width, self.segment_count)
}

///|
pub fn GreedySegmentPlan::workspace_bytes(self : GreedySegmentPlan) -> Int64 {
  self.workspace_bytes
}
