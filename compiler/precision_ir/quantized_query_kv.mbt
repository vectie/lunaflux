///|
/// The unweighted head normalization preserves each BF16 expression boundary,
/// unlike learned RMSNorm, whose reduction and affine arithmetic stay F32.
pub(all) enum QueryHeadNormalization {
  Bf16SquareMeanEpsilonInverseProduct
} derive(Debug, Eq)

///|
pub extend QueryHeadNormalization with Eq::{equal, not_equal}

///|
pub extend QueryHeadNormalization with @debug.Debug::{to_repr}

///|
/// Model-neutral low-rank query and shared-KV projection prefix. Rotary,
/// cache quantization, compression and attention are subsequent operations.
pub struct QuantizedQueryKvPrecision {
  priv hidden : Int
  priv rank : Int
  priv heads : Int
  priv width : Int
  priv rows : Int
  priv epsilon : Double
  priv activation_quantization : QuantizationPlan
  priv representations : ReadOnlyArray[@format.Representation]
  priv frames : ReadOnlyArray[Int64]
  priv weights : ReadOnlyArray[Int64]
  priv workspace : Int64
}

///|
pub fn QuantizedQueryKvPrecision::new(
  hidden_width~ : Int,
  query_rank~ : Int,
  head_count~ : Int,
  head_width~ : Int,
  maximum_rows~ : Int,
  normalization_epsilon~ : Double,
  activation_amax_floor~ : Double,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  guard hidden_width > 0 &&
    hidden_width <= 65536 &&
    query_rank > 0 &&
    query_rank <= 65536 &&
    head_count > 0 &&
    head_count <= 512 &&
    head_width > 0 &&
    head_width <= 65536 &&
    head_count.to_int64() * head_width.to_int64() <= 65536L &&
    maximum_rows > 0 &&
    maximum_rows <= 65536 &&
    hidden_width % 128 == 0 &&
    query_rank % 128 == 0 &&
    head_width % 128 == 0 &&
    normalization_epsilon > 0.0 &&
    normalization_epsilon <= 1.0 &&
    !normalization_epsilon.is_nan() &&
    !normalization_epsilon.is_inf() else {
    raise PrecisionPlanError::IllegalPlacement
  }
  let matrix = (input, output) => {
    @format.Representation::new(
      format=MxFp8E4M3,
      rows=output,
      columns=input,
      grouping=Block(rows=128, columns=128),
    )
  }
  let representations = [
    matrix(hidden_width, query_rank),
    matrix(query_rank, head_count * head_width),
    matrix(hidden_width, head_width),
  ]
  let activation_quantization = plan_operand(
      role=Activation,
      representation=@format.Representation::new(
        format=MxFp8E4M3,
        rows=maximum_rows,
        columns=hidden_width,
        grouping=Block(rows=1, columns=128),
      ),
      placement=QuantizeBeforeConsumer,
      supported_placements=[QuantizeBeforeConsumer],
      accuracy=Accuracy::new(absolute=0.0, relative=0.0),
      memory_ceiling=9223372036854775807L,
    )
    .quantization()
    .with_amax_floor(activation_amax_floor)
  let row_bytes = (width : Int) => {
    expert_checked_mul(
      expert_checked_mul(maximum_rows.to_int64(), width.to_int64()),
      2L,
    )
  }
  let frames = [
    row_bytes(query_rank),
    row_bytes(query_rank),
    row_bytes(head_count * head_width),
    row_bytes(head_count * head_width),
    row_bytes(head_width),
    row_bytes(head_width),
  ]
  let workspace = frames.fold(init=0L, (sum, bytes) => sum + bytes)
  guard workspace <= maximum_workspace_bytes else {
    raise PrecisionPlanError::MemoryBudgetExceeded
  }
  let weights = [
    representations[0].payload_bytes(),
    representations[0].scale_bytes(),
    query_rank.to_int64() * 2L,
    representations[1].payload_bytes(),
    representations[1].scale_bytes(),
    representations[2].payload_bytes(),
    representations[2].scale_bytes(),
    head_width.to_int64() * 2L,
  ]
  {
    hidden: hidden_width,
    rank: query_rank,
    heads: head_count,
    width: head_width,
    rows: maximum_rows,
    epsilon: normalization_epsilon,
    activation_quantization,
    representations: ReadOnlyArray::from_array(representations),
    frames: ReadOnlyArray::from_array(frames),
    weights: ReadOnlyArray::from_array(weights),
    workspace,
  }
}

///|
pub fn QuantizedQueryKvPrecision::geometry(
  self : Self,
) -> (Int, Int, Int, Int, Int) {
  (self.hidden, self.rank, self.heads, self.width, self.rows)
}

///|
pub fn QuantizedQueryKvPrecision::normalization_epsilon(self : Self) -> Double {
  self.epsilon
}

///|
pub fn QuantizedQueryKvPrecision::head_normalization(
  _self : Self,
) -> QueryHeadNormalization {
  Bf16SquareMeanEpsilonInverseProduct
}

///|
pub fn QuantizedQueryKvPrecision::activation_quantization(
  self : Self,
) -> QuantizationPlan {
  self.activation_quantization
}

///|
/// Query-A, Query-B, shared KV; each remains a packed E4M3/UE8M0 matrix.
pub fn QuantizedQueryKvPrecision::weight_representations(
  self : Self,
) -> ReadOnlyArray[@format.Representation] {
  self.representations
}

///|
pub fn QuantizedQueryKvPrecision::frame_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.frames
}

///|
/// Query-A payload/scale/norm, Query-B payload/scale, KV payload/scale/norm.
pub fn QuantizedQueryKvPrecision::weight_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.weights
}

///|
pub fn QuantizedQueryKvPrecision::workspace_bytes(self : Self) -> Int64 {
  self.workspace
}
