///|
fn hash_operand_plan_field(hash : @crypto.SHA256, value : StringView) -> Unit {
  let prefix = @utf8.encode(value.length().to_string() + ":")
  let encoded = @utf8.encode(value)
  hash.update_from_iter(prefix.iter())
  hash.update_from_iter(encoded.iter())
  hash.update_from_iter([b'\n'].iter())
}

///|
fn canonical_operand_plan_digest(
  plan : DeepSeekV4TokenHashOperandPlan,
) -> DeepSeekV4TokenHashOperandPlanDigest {
  let hash = @crypto.SHA256::new()
  hash_operand_plan_field(
    hash, "lunaflux/deepseek-v4-token-hash-operand-plan/v1",
  )
  hash_operand_plan_field(hash, plan.source_identity.content().as_hex())
  hash_operand_plan_field(hash, plan.source_identity.plan().as_hex())
  hash_operand_plan_field(hash, plan.requirements_digest.as_hex())
  hash_operand_plan_field(hash, plan.candidate_source_digest.as_hex())
  hash_operand_plan_field(hash, plan.candidate_recipe_digest.as_hex())
  hash_operand_plan_field(hash, plan.sidecar_layout_digest.as_hex())
  hash_operand_plan_field(hash, plan.sidecar_source_digest.as_hex())
  hash_operand_plan_field(hash, plan.source_manifest_digest)
  hash_operand_plan_field(hash, plan.table_source_label)
  hash_operand_plan_field(hash, plan.requirement_ordinal.to_string())
  hash_operand_plan_field(hash, plan.layer_index.to_string())
  hash_operand_plan_field(hash, plan.routed_expert_count.to_string())
  hash_operand_plan_field(hash, plan.operand_ordinal.to_string())
  hash_operand_plan_field(hash, "token-hash-table-input")
  hash_operand_plan_field(hash, plan.byte_count.to_string())
  hash_operand_plan_field(hash, plan.alignment.to_string())
  hash_operand_plan_field(hash, plan.region_ordinal.to_string())
  hash_operand_plan_field(hash, plan.region_tensor_ordinal.to_string())
  hash_operand_plan_field(hash, plan.device_offset.to_string())
  DeepSeekV4TokenHashOperandPlanDigest(
    @crypto.bytes_to_hex_string(hash.finalize()),
  )
}
