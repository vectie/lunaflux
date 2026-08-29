///|
fn fused_qualification_residual_reference(
  shape : FusedQualificationShape,
  residual : ArrayView[Int],
  branch : ArrayView[Int],
  norm_weight : ArrayView[Int],
) -> (Array[Int], Array[Int]) raise FusedQualificationError {
  let elements = shape.token_count * FUSED_QUALIFICATION_INPUT_WIDTH
  if residual.length() != elements ||
    branch.length() != elements ||
    norm_weight.length() != FUSED_QUALIFICATION_INPUT_WIDTH {
    raise InvalidRefereeInput
  }
  let residual_output = Array::make(elements, FUSED_QUALIFICATION_POISON)
  let norm_output = Array::make(elements, FUSED_QUALIFICATION_POISON)
  let partial = FixedArray::make(FUSED_QUALIFICATION_BLOCK_THREADS, 0.0F)
  for token in 0..<shape.token_count {
    for lane in 0..<FUSED_QUALIFICATION_BLOCK_THREADS {
      let offset = token * FUSED_QUALIFICATION_INPUT_WIDTH + lane
      let value = fused_qualification_bf16_float(residual[offset]) +
        fused_qualification_bf16_float(branch[offset])
      let encoded = fused_qualification_bf16_bits(value)
      residual_output[offset] = encoded
      let rounded = fused_qualification_bf16_float(encoded)
      partial[lane] = rounded * rounded
    }
    for stride = FUSED_QUALIFICATION_BLOCK_THREADS / 2
        stride > 0
        stride = stride / 2 {
      for lane in 0..<stride {
        partial[lane] += partial[lane + stride]
      }
    }
    let inverse_rms = 1.0F /
      (partial[0] / Float::from_int(FUSED_QUALIFICATION_INPUT_WIDTH) + 0.00001F).sqrt()
    for column in 0..<FUSED_QUALIFICATION_INPUT_WIDTH {
      let offset = token * FUSED_QUALIFICATION_INPUT_WIDTH + column
      norm_output[offset] = fused_qualification_bf16_bits(
        fused_qualification_bf16_float(residual_output[offset]) *
        inverse_rms *
        fused_qualification_bf16_float(norm_weight[column]),
      )
    }
  }
  (residual_output, norm_output)
}
