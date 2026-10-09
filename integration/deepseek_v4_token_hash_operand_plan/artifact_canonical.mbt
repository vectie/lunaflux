///|
fn canonical_artifact_binding_digest(
  binding : DeepSeekV4TokenHashArtifactBinding,
) -> DeepSeekV4TokenHashArtifactBindingDigest {
  let hash = @crypto.SHA256::new()
  hash_operand_plan_field(
    hash, "lunaflux/deepseek-v4-token-hash-artifact-binding/v1",
  )
  hash_operand_plan_field(hash, binding.source_identity.content().as_hex())
  hash_operand_plan_field(hash, binding.source_identity.plan().as_hex())
  hash_operand_plan_field(hash, binding.requirements_digest.as_hex())
  hash_operand_plan_field(hash, binding.layer_index.to_string())
  hash_operand_plan_field(hash, binding.requirement_ordinal.to_string())
  hash_operand_plan_field(hash, binding.operand_plan_digest.as_hex())
  hash_operand_plan_field(hash, binding.candidate_source_digest.as_hex())
  hash_operand_plan_field(hash, binding.candidate_recipe_digest.as_hex())
  hash_operand_plan_field(hash, binding.artifact_admission_digest.as_hex())
  hash_operand_plan_field(hash, binding.target.compute_major().to_string())
  hash_operand_plan_field(hash, binding.target.compute_minor().to_string())
  hash_operand_plan_field(hash, binding.target.supports_bf16().to_string())
  hash_operand_plan_field(hash, binding.target.supports_cublas_lt().to_string())
  hash_operand_plan_field(hash, binding.module_digest.sha256())
  hash_operand_plan_field(hash, binding.module_byte_count.to_string())
  hash_operand_plan_field(
    hash,
    binding.entry_point.family().family_id().as_int().to_string(),
  )
  hash_operand_plan_field(
    hash,
    binding.entry_point.entry_point_id().as_int().to_string(),
  )
  hash_operand_plan_field(hash, binding.function_symbol)
  let dimensions = binding.dimensions
  for
    value in [
      dimensions.grid_x(),
      dimensions.grid_y(),
      dimensions.grid_z(),
      dimensions.block_x(),
      dimensions.block_y(),
      dimensions.block_z(),
      dimensions.shared_memory_bytes(),
      binding.operand_count,
      binding.token_hash_operand_ordinal,
    ] {
    hash_operand_plan_field(hash, value.to_string())
  }
  hash_operand_plan_field(hash, binding.token_hash_device_offset.to_string())
  DeepSeekV4TokenHashArtifactBindingDigest(
    @crypto.bytes_to_hex_string(hash.finalize()),
  )
}
