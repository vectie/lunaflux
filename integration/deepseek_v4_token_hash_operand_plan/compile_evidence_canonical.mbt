///|
fn canonical_compile_evidence_binding_digest(
  binding : DeepSeekV4TokenHashCompileEvidenceBinding,
) -> DeepSeekV4TokenHashCompileEvidenceBindingDigest {
  let hash = @crypto.SHA256::new()
  hash_operand_plan_field(
    hash, "lunaflux/deepseek-v4-token-hash-compile-evidence-binding/v1",
  )
  hash_operand_plan_field(hash, binding.source_identity.content().as_hex())
  hash_operand_plan_field(hash, binding.source_identity.plan().as_hex())
  hash_operand_plan_field(hash, binding.requirements_digest.as_hex())
  hash_operand_plan_field(hash, binding.layer_index.to_string())
  hash_operand_plan_field(hash, binding.requirement_ordinal.to_string())
  hash_operand_plan_field(hash, binding.artifact_binding_digest.as_hex())
  hash_operand_plan_field(hash, binding.compile_evidence_digest.as_hex())
  hash_operand_plan_field(hash, binding.source_digest_sha256)
  hash_operand_plan_field(hash, binding.recipe_digest_sha256)
  hash_operand_plan_field(hash, binding.driver_identity_sha256)
  hash_operand_plan_field(hash, binding.target.compute_major().to_string())
  hash_operand_plan_field(hash, binding.target.compute_minor().to_string())
  hash_operand_plan_field(hash, binding.target.supports_bf16().to_string())
  hash_operand_plan_field(hash, binding.target.supports_cublas_lt().to_string())
  hash_operand_plan_field(hash, binding.function_symbol)
  hash_operand_plan_field(hash, binding.module_digest.sha256())
  hash_operand_plan_field(hash, binding.module_byte_count.to_string())
  DeepSeekV4TokenHashCompileEvidenceBindingDigest(
    @crypto.bytes_to_hex_string(hash.finalize()),
  )
}
