///|
/// A BF16 residual [row, stream, hidden] is reduced in F32, rounded once,
/// and published into one segment of [row, segment, hidden]. Segment order
/// is semantic; GPU geometry and model layer numbers are not part of this IR.
pub struct StreamMeanCapturePrecision {
  priv streams : Int
  priv hidden : Int
  priv rows : Int
  priv segments : Int
  priv input_bytes : Int64
  priv output_bytes : Int64
}

///|
pub fn StreamMeanCapturePrecision::new(
  stream_count~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  segment_count~ : Int,
  maximum_output_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard stream_count > 0 &&
    hidden_width > 0 &&
    maximum_rows > 0 &&
    segment_count > 0 else {
    raise IllegalPlacement
  }
  let vector = expert_checked_mul(
    maximum_rows.to_int64(),
    hidden_width.to_int64(),
  )
  let input_bytes = expert_checked_mul(
    expert_checked_mul(vector, stream_count.to_int64()),
    2L,
  )
  let output_bytes = expert_checked_mul(
    expert_checked_mul(vector, segment_count.to_int64()),
    2L,
  )
  guard output_bytes <= maximum_output_bytes else { raise MemoryBudgetExceeded }
  {
    streams: stream_count,
    hidden: hidden_width,
    rows: maximum_rows,
    segments: segment_count,
    input_bytes,
    output_bytes,
  }
}

///|
pub fn StreamMeanCapturePrecision::geometry(
  self : Self,
) -> (Int, Int, Int, Int) {
  (self.streams, self.hidden, self.rows, self.segments)
}

///|
pub fn StreamMeanCapturePrecision::input_bytes(self : Self) -> Int64 {
  self.input_bytes
}

///|
pub fn StreamMeanCapturePrecision::output_bytes(self : Self) -> Int64 {
  self.output_bytes
}
