///|
fn normalize_packed_head(
  input : ReadOnlyArray[Float],
  output : Array[Float],
  row_offset : Int,
  head_offset : Int,
  head_dimension : Int,
  weight : @refkernels.Tensor,
  epsilon : Float,
) -> Unit raise @refkernels.ReferenceKernelError {
  let mut square_sum = 0.0F
  for column in 0..<head_dimension {
    let value = input[row_offset + head_offset + column]
    square_sum += value * value
  }
  if square_sum.is_nan() || square_sum.is_inf() {
    raise NonFinite(ComputedValue)
  }
  let inverse_rms = 1.0F /
    (square_sum / Float::from_int(head_dimension) + epsilon).sqrt()
  for column in 0..<head_dimension {
    output[row_offset + head_offset + column] = input[row_offset +
      head_offset +
      column] *
      inverse_rms *
      weight.values()[column]
  }
}

///|
/// Applies Qwen3 independent per-token, per-head RMS normalization to the Q
/// and K portions of one packed Q/K/V activation. V is copied exactly.
fn qk_rms_norm(
  input : @refkernels.Activation,
  query_weight : @refkernels.Tensor,
  key_weight : @refkernels.Tensor,
  query_heads~ : Int,
  kv_heads~ : Int,
  head_dimension~ : Int,
  epsilon~ : Float,
  limits~ : @refkernels.ReferenceLimits,
) -> @refkernels.Activation raise @refkernels.ReferenceKernelError {
  if query_heads <= 0 || kv_heads <= 0 || head_dimension <= 0 {
    raise InvalidShape(PositiveDimensions)
  }
  if kv_heads > (2147483647 - query_heads) / 2 {
    raise ArithmeticOverflow
  }
  let packed_heads = query_heads + 2 * kv_heads
  if packed_heads > 2147483647 / head_dimension {
    raise ArithmeticOverflow
  }
  let packed_width = packed_heads * head_dimension
  if input.columns() != packed_width ||
    query_weight.rows() != 1 ||
    query_weight.columns() != head_dimension ||
    key_weight.rows() != 1 ||
    key_weight.columns() != head_dimension {
    raise InvalidShape(NormalizationWeight)
  }
  if epsilon.is_nan() || epsilon.is_inf() {
    raise NonFinite(Epsilon)
  }
  if epsilon <= 0.0F {
    raise InvalidScalar(Epsilon)
  }
  let output : Array[Float] = []
  for value in input.values() {
    output.push(value)
  }
  let query_width = query_heads * head_dimension
  for row in 0..<input.rows() {
    let row_offset = row * packed_width
    for head in 0..<query_heads {
      normalize_packed_head(
        input.values(),
        output,
        row_offset,
        head * head_dimension,
        head_dimension,
        query_weight,
        epsilon,
      )
    }
    for head in 0..<kv_heads {
      normalize_packed_head(
        input.values(),
        output,
        row_offset,
        query_width + head * head_dimension,
        head_dimension,
        key_weight,
        epsilon,
      )
    }
  }
  @refkernels.Activation::new(
    rows=input.rows(),
    columns=packed_width,
    values=output,
    limits~,
  )
}
