///|
fn exact_dsa_kv_attention_assembly_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  head_count : Int,
  key_nope_width : Int,
  rope_width : Int,
  value_width : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if head_count <= 0 ||
    key_nope_width <= 0 ||
    rope_width < 0 ||
    value_width <= 0 {
    return false
  }
  let (counts, packed, suffix, keys, values) = if rope_width > 0 {
    guard claim.operands is [counts, packed, suffix, keys, values] else {
      return false
    }
    (counts, packed, Some(suffix), keys, values)
  } else {
    guard claim.operands is [counts, packed, keys, values] else { return false }
    (counts, packed, None, keys, values)
  }
  let heads = head_count.to_int64()
  let packed_width = (key_nope_width + value_width).to_int64()
  let key_width = (key_nope_width + rope_width).to_int64()
  guard checked_positive_mul(heads, packed_width) is Some(packed_row_cells) &&
    checked_positive_mul(packed_row_cells, 2L) is Some(packed_row_bytes) &&
    exact_rows(packed.byte_count, packed_row_bytes) is Some(rows) &&
    checked_positive_mul(heads, key_width) is Some(key_row_cells) &&
    checked_positive_mul(key_row_cells, 2L) is Some(key_row_bytes) &&
    checked_positive_mul(heads, value_width.to_int64()) is Some(value_row_cells) &&
    checked_positive_mul(value_row_cells, 2L) is Some(value_row_bytes) &&
    checked_positive_mul(rows, key_row_bytes) is Some(key_bytes) &&
    checked_positive_mul(rows, value_row_bytes) is Some(value_bytes) &&
    checked_positive_mul(rows, key_row_cells + value_row_cells) is Some(work) else {
    return false
  }
  let suffix_exact = match suffix {
    Some(item) =>
      checked_positive_mul(rows, rope_width.to_int64()) is Some(suffix_cells) &&
      operand_is(item, DsaAttentionKeyRopeInput, suffix_cells * 2L, 16L)
    None => rope_width == 0
  }
  let expected_symbol = if rope_width > 0 {
    "lunaflux_glm_hybrid_dsa_kv_assembly_rope_bf16_v1"
  } else {
    "lunaflux_glm_hybrid_dsa_kv_assembly_no_rope_bf16_v1"
  }
  rows <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol == expected_symbol &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(packed, DsaKeyValueHeadsOutput, packed.byte_count, 16L) &&
  suffix_exact &&
  operand_is(keys, DsaAttentionKeysInput, key_bytes, 16L) &&
  operand_is(values, DsaAttentionValuesInput, value_bytes, 16L) &&
  exact_launch(claim.launch, work, 0)
}
