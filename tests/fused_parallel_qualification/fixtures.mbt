///|
fn fused_qualification_value(index : Int, phase : Int) -> Float {
  Float::from_int((index * (phase * 2 + 1) + phase * 7) % 29 - 14) * 0.0078125F
}

///|
fn fused_qualification_values(length : Int, phase : Int) -> Array[Int] {
  Array::makei(length, index => {
    fused_qualification_bf16_bits(fused_qualification_value(index, phase))
  })
}

///|
fn fused_qualification_norm_weight() -> Array[Int] {
  Array::makei(FUSED_QUALIFICATION_INPUT_WIDTH, index => {
    fused_qualification_bf16_bits(0.75F + Float::from_int(index % 9) * 0.03125F)
  })
}

///|
pub fn qkv_qualification_fixture(
  shape : FusedQualificationShape,
) -> FusedQkvQualificationFixture raise FusedQualificationError {
  let input = fused_qualification_values(
    shape.token_count * FUSED_QUALIFICATION_INPUT_WIDTH,
    1,
  )
  let q_weight = fused_qualification_values(
    FUSED_QUALIFICATION_INPUT_WIDTH * FUSED_QUALIFICATION_QUERY_WIDTH,
    2,
  )
  let k_weight = fused_qualification_values(
    FUSED_QUALIFICATION_INPUT_WIDTH * FUSED_QUALIFICATION_KV_WIDTH,
    3,
  )
  let v_weight = fused_qualification_values(
    FUSED_QUALIFICATION_INPUT_WIDTH * FUSED_QUALIFICATION_KV_WIDTH,
    4,
  )
  let (rotated_qkv, key_cache, value_cache) = fused_qualification_qkv_reference(
    shape, input, q_weight, k_weight, v_weight,
  )
  {
    shape,
    input: ReadOnlyArray::from_array(input),
    q_weight: ReadOnlyArray::from_array(q_weight),
    k_weight: ReadOnlyArray::from_array(k_weight),
    v_weight: ReadOnlyArray::from_array(v_weight),
    rotated_qkv: ReadOnlyArray::from_array(rotated_qkv),
    key_cache: ReadOnlyArray::from_array(key_cache),
    value_cache: ReadOnlyArray::from_array(value_cache),
  }
}

///|
pub fn residual_norm_qualification_fixture(
  shape : FusedQualificationShape,
) -> FusedResidualNormQualificationFixture raise FusedQualificationError {
  let residual = fused_qualification_values(
    shape.token_count * FUSED_QUALIFICATION_INPUT_WIDTH,
    5,
  )
  let branch = fused_qualification_values(
    shape.token_count * FUSED_QUALIFICATION_INPUT_WIDTH,
    6,
  )
  let norm_weight = fused_qualification_norm_weight()
  let (residual_output, norm_output) = fused_qualification_residual_reference(
    shape, residual, branch, norm_weight,
  )
  {
    shape,
    residual: ReadOnlyArray::from_array(residual),
    branch: ReadOnlyArray::from_array(branch),
    norm_weight: ReadOnlyArray::from_array(norm_weight),
    residual_output: ReadOnlyArray::from_array(residual_output),
    norm_output: ReadOnlyArray::from_array(norm_output),
  }
}
