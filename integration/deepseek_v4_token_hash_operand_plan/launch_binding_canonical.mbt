///|
fn canonical_token_hash_launch_binding_digest(
  binding : DeepSeekV4TokenHashLaunchBinding,
) -> DeepSeekV4TokenHashLaunchBindingDigest {
  let hash = @crypto.SHA256::new()
  hash_operand_plan_field(
    hash, "lunaflux/deepseek-v4-token-hash-launch-binding/v1",
  )
  hash_operand_plan_field(hash, binding.source_identity.content().as_hex())
  hash_operand_plan_field(hash, binding.source_identity.plan().as_hex())
  hash_operand_plan_field(hash, binding.requirements_digest.as_hex())
  hash_operand_plan_field(hash, binding.layer_index.to_string())
  hash_operand_plan_field(hash, binding.requirement_ordinal.to_string())
  hash_operand_plan_field(hash, binding.maximum_tokens.to_string())
  hash_operand_plan_field(hash, binding.experts_per_token.to_string())
  hash_operand_plan_field(hash, binding.routed_expert_count.to_string())
  hash_operand_plan_field(hash, binding.operand_plan_digest.as_hex())
  hash_operand_plan_field(hash, binding.sidecar_layout_digest.as_hex())
  hash_operand_plan_field(hash, binding.artifact_binding_digest.as_hex())
  hash_operand_plan_field(hash, binding.compile_binding_digest.as_hex())
  hash_operand_plan_field(hash, binding.compile_evidence_digest.as_hex())
  hash_operand_plan_field(hash, binding.operand_regions.digest().as_hex())
  hash_operand_plan_field(hash, binding.source_digest_sha256)
  hash_operand_plan_field(hash, binding.recipe_digest_sha256)
  let compiler_identity = binding.compiler.identity()
  hash_operand_plan_field(hash, compiler_identity.toolchain_digest_sha256())
  hash_operand_plan_field(hash, compiler_identity.major().to_string())
  hash_operand_plan_field(hash, compiler_identity.minor().to_string())
  hash_operand_plan_field(hash, compiler_identity.patch().to_string())
  hash_operand_plan_field(
    hash,
    binding.compiler.optimization_level().to_string(),
  )
  hash_operand_plan_field(
    hash,
    binding.compiler.fused_multiply_add().to_string(),
  )
  hash_operand_plan_field(hash, binding.compiler.reassociate().to_string())
  hash_operand_plan_field(hash, binding.compiler.max_registers().to_string())
  hash_operand_plan_field(hash, binding.target.compute_major().to_string())
  hash_operand_plan_field(hash, binding.target.compute_minor().to_string())
  hash_operand_plan_field(hash, binding.target.supports_bf16().to_string())
  hash_operand_plan_field(hash, binding.target.supports_cublas_lt().to_string())
  hash_operand_plan_field(
    hash,
    binding.entry_point.family().module_digest().sha256(),
  )
  hash_operand_plan_field(
    hash,
    binding.entry_point.family().family_id().as_int().to_string(),
  )
  hash_operand_plan_field(
    hash,
    binding.entry_point.entry_point_id().as_int().to_string(),
  )
  hash_operand_plan_field(hash, binding.function_symbol)
  hash_operand_plan_field(hash, binding.module_digest.sha256())
  for
    value in [
      binding.dimensions.grid_x(),
      binding.dimensions.grid_y(),
      binding.dimensions.grid_z(),
      binding.dimensions.block_x(),
      binding.dimensions.block_y(),
      binding.dimensions.block_z(),
      binding.dimensions.shared_memory_bytes(),
    ] {
    hash_operand_plan_field(hash, value.to_string())
  }
  DeepSeekV4TokenHashLaunchBindingDigest(
    @crypto.bytes_to_hex_string(hash.finalize()),
  )
}
