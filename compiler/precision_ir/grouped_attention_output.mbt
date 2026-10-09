///|
/// Derotate head outputs, group-local low-rank projection with BF16 parameters,
/// then block-FP8 projection back to hidden width. Output-A's compact parameter
/// dequantization has a BF16 boundary; Output-B quantizes activations per block.
pub struct GroupedAttentionOutputPrecision {
  priv rotary : RotaryTensorPrecision
  priv groups : Int
  priv rank : Int
  priv hidden : Int
  priv matrices : ReadOnlyArray[@format.Representation]
  priv frames : ReadOnlyArray[Int64]
}

///|
pub fn GroupedAttentionOutputPrecision::new(
  head_count~ : Int,
  head_width~ : Int,
  rotary_width~ : Int,
  group_count~ : Int,
  output_rank~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  maximum_positions~ : Int,
  frequency~ : RotaryFrequency,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  guard group_count > 0 &&
    group_count <= head_count &&
    head_count % group_count == 0 &&
    output_rank > 0 &&
    output_rank <= 65536 &&
    output_rank % 128 == 0 &&
    hidden_width > 0 &&
    hidden_width <= 65536 &&
    hidden_width % 128 == 0 &&
    group_count.to_int64() * output_rank.to_int64() <= 65536L &&
    head_count.to_int64() *
    head_width.to_int64() /
    group_count.to_int64() %
    128L ==
    0L else {
    raise PrecisionPlanError::IllegalPlacement
  }
  let rotary = RotaryTensorPrecision::new(
    head_count~,
    head_width~,
    rotary_width~,
    maximum_rows~,
    maximum_positions~,
    frequency~,
    inverse=true,
    maximum_workspace_bytes~,
  )
  let frames = [
    rotary.workspace_bytes(),
    maximum_rows.to_int64() *
    group_count.to_int64() *
    output_rank.to_int64() *
    2L,
    maximum_rows.to_int64() * hidden_width.to_int64() * 2L,
  ]
  guard frames.fold(init=0L, (total, bytes) => total + bytes) <=
    maximum_workspace_bytes else {
    raise PrecisionPlanError::MemoryBudgetExceeded
  }
  let matrix = (rows, columns) => {
    @format.Representation::new(
      format=MxFp8E4M3,
      rows~,
      columns~,
      grouping=Block(rows=128, columns=128),
    )
  }
  {
    rotary,
    groups: group_count,
    rank: output_rank,
    hidden: hidden_width,
    matrices: ReadOnlyArray::from_array([
      matrix(group_count * output_rank, head_count * head_width / group_count),
      matrix(hidden_width, group_count * output_rank),
    ]),
    frames: ReadOnlyArray::from_array(frames),
  }
}

///|
pub fn GroupedAttentionOutputPrecision::rotary(
  self : Self,
) -> RotaryTensorPrecision {
  self.rotary
}

///|
pub fn GroupedAttentionOutputPrecision::geometry(
  self : Self,
) -> (Int, Int, Int, Int) {
  let (heads, width, _, rows, _) = self.rotary.geometry()
  (self.groups, heads * width / self.groups, self.rank, rows)
}

///|
pub fn GroupedAttentionOutputPrecision::hidden_width(self : Self) -> Int {
  self.hidden
}

///|
pub fn GroupedAttentionOutputPrecision::weight_representations(
  self : Self,
) -> ReadOnlyArray[@format.Representation] {
  self.matrices
}

///|
pub fn GroupedAttentionOutputPrecision::frame_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.frames
}

///|
pub fn GroupedAttentionOutputPrecision::workspace_bytes(self : Self) -> Int64 {
  self.frames.fold(init=0L, (total, bytes) => total + bytes)
}
