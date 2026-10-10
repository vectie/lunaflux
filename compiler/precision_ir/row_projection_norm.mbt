///|
/// Packed E4M3/UE8M0 row projection followed by learned RMSNorm. Projection
/// rounds to BF16 before the F32 normalization reduction/affine and final BF16
/// round. This semantic boundary cannot be removed by fusion without an
/// explicit alternative accuracy contract.
pub struct RowProjectionNormPrecision {
  priv input : Int
  priv output : Int
  priv rows : Int
  priv epsilon : Double
  priv matrix : @format.Representation
  priv quantization : QuantizationPlan
  priv frame_bytes : Int64
}

///|
pub fn RowProjectionNormPrecision::new(
  input_width~ : Int,
  output_width~ : Int,
  maximum_rows~ : Int,
  normalization_epsilon~ : Double,
  activation_amax_floor~ : Double,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  guard input_width > 0 &&
    input_width <= 65536 &&
    input_width % 128 == 0 &&
    output_width > 0 &&
    output_width <= 65536 &&
    output_width % 128 == 0 &&
    maximum_rows > 0 &&
    maximum_rows <= 65536 &&
    normalization_epsilon > 0.0 &&
    normalization_epsilon <= 1.0 &&
    !normalization_epsilon.is_nan() &&
    !normalization_epsilon.is_inf() else {
    raise PrecisionPlanError::IllegalPlacement
  }
  let frame_bytes = expert_checked_mul(
    expert_checked_mul(maximum_rows.to_int64(), output_width.to_int64()),
    2L,
  )
  guard expert_checked_mul(frame_bytes, 2L) <= maximum_workspace_bytes else {
    raise PrecisionPlanError::MemoryBudgetExceeded
  }
  let quantization = plan_operand(
      role=Activation,
      representation=@format.Representation::new(
        format=MxFp8E4M3,
        rows=maximum_rows,
        columns=input_width,
        grouping=Block(rows=1, columns=128),
      ),
      placement=QuantizeBeforeConsumer,
      supported_placements=[QuantizeBeforeConsumer],
      accuracy=Accuracy::new(absolute=0.0, relative=0.0),
      memory_ceiling=9223372036854775807L,
    )
    .quantization()
    .with_amax_floor(activation_amax_floor)
  {
    input: input_width,
    output: output_width,
    rows: maximum_rows,
    epsilon: normalization_epsilon,
    matrix: @format.Representation::new(
      format=MxFp8E4M3,
      rows=output_width,
      columns=input_width,
      grouping=Block(rows=128, columns=128),
    ),
    quantization,
    frame_bytes,
  }
}

///|
pub fn RowProjectionNormPrecision::geometry(self : Self) -> (Int, Int, Int) {
  (self.input, self.output, self.rows)
}

///|
pub fn RowProjectionNormPrecision::normalization_epsilon(self : Self) -> Double {
  self.epsilon
}

///|
pub fn RowProjectionNormPrecision::activation_quantization(
  self : Self,
) -> QuantizationPlan {
  self.quantization
}

///|
pub fn RowProjectionNormPrecision::weight_representation(
  self : Self,
) -> @format.Representation {
  self.matrix
}

///|
pub fn RowProjectionNormPrecision::frame_bytes(self : Self) -> Int64 {
  self.frame_bytes
}

///|
pub fn RowProjectionNormPrecision::workspace_bytes(self : Self) -> Int64 {
  self.frame_bytes * 2L
}

///|
pub fn RowProjectionNormPrecision::input_bytes(self : Self) -> Int64 {
  self.rows.to_int64() * self.input.to_int64() * 2L
}
