///|
fn postprocess_bool(value : Bool) -> String {
  if value {
    "true"
  } else {
    "false"
  }
}

///|
fn postprocess_operand_record(
  operands : ArrayView[@launch_contract.AotOperand],
) -> String {
  let output = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      output.write_string(";")
    }
    output.write_string(
      index.to_string() +
      ":" +
      operand.byte_count().to_string() +
      ":" +
      operand.alignment().to_string(),
    )
  }
  output.to_string()
}

///|
fn postprocess_recipe(
  plan : @plan.ModelPlan,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  query_norm_weight : @plan.TensorRef,
  key_norm_weight : @plan.TensorRef,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  alias_contract : QkNormRopePagedKvWriteAliasContract,
  source_digest : QkNormRopePagedKvWriteDigest,
) -> Bytes {
  let (qkv_operation, norm_operation, rope_operation, attention_operation) = semantic.operation_indices()
  let (_, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-qknorm-rope-paged-kv-write-candidate.v2\n" +
    "family=qknorm-positioned-rope-paged-kv-write\n" +
    "execution_policy=projection-postprocess-production\n" +
    "projection_included=false\nprojection_weights_read=false\ngemm_included=false\n" +
    "model_content_sha256=" +
    plan.identity().content().as_hex() +
    "\n" +
    "model_plan_sha256=" +
    plan.identity().plan().as_hex() +
    "\n" +
    "qkv_operation=" +
    qkv_operation.to_string() +
    "\n" +
    "qk_norm_operation=" +
    norm_operation.to_string() +
    "\n" +
    "rope_operation=" +
    rope_operation.to_string() +
    "\n" +
    "attention_operation=" +
    attention_operation.to_string() +
    "\n" +
    "projected_qkv_activation=" +
    semantic.projected_qkv().as_int().to_string() +
    "\n" +
    "rotated_qkv_activation=" +
    semantic.positioned_qkv().as_int().to_string() +
    "\n" +
    "query_norm_weight=" +
    query_norm_weight.as_int().to_string() +
    "\n" +
    "key_norm_weight=" +
    key_norm_weight.as_int().to_string() +
    "\n" +
    "kv_layer=" +
    semantic.kv_layer().as_int().to_string() +
    "\n" +
    "shape=" +
    output_width.to_string() +
    "," +
    query_heads.to_string() +
    "," +
    kv_heads.to_string() +
    "," +
    head_dimension.to_string() +
    "\n" +
    "norm_epsilon=" +
    semantic.norm_epsilon().to_string() +
    "\n" +
    "rope_theta=" +
    semantic.rope_theta().to_string() +
    "\n" +
    "layout=" +
    layout.stable_layout_version().to_string() +
    "," +
    layout.tokens_per_page().to_string() +
    "," +
    layout.total_page_count().to_string() +
    "," +
    layout.page_stride_bytes().to_string() +
    "\n" +
    "profile=" +
    profile.max_query_rows().to_string() +
    "," +
    profile.max_query_tokens().to_string() +
    "," +
    profile.max_page_table_entries().to_string() +
    "\n" +
    "target=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\n" +
    "toolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\n" +
    "compiler=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\n" +
    "optimization=" +
    compiler.optimization_level().to_string() +
    "\n" +
    "fmad=" +
    postprocess_bool(compiler.fused_multiply_add()) +
    "\n" +
    "reassociate=" +
    postprocess_bool(compiler.reassociate()) +
    "\n" +
    "max_registers=" +
    compiler.max_registers().to_string() +
    "\n" +
    "symbol=" +
    postprocess_production_symbol +
    "\n" +
    "abi=" +
    QkNormRopePagedKvWriteKernelAbi::QwenQkNormPositionedRopePagedKvWriteProductionV2.canonical_name() +
    "\n" +
    "grid=" +
    dimensions.grid_x().to_string() +
    "," +
    dimensions.grid_y().to_string() +
    ",1\n" +
    "grid_x_policy=query-token-capped\ngrid_stride_tokens=true\nblock=128,1,1\nshared_memory_bytes=0\n" +
    "alias_contract=" +
    alias_contract.canonical_name() +
    "\n" +
    "operands=" +
    postprocess_operand_record(operands) +
    "\n" +
    "source_sha256=" +
    source_digest.as_hex() +
    "\n" +
    "aot_only=true\nruntime_jit=false\n" +
    "physical_qualification=required\nexternal_approval=required\n" +
    "manifest_bindable=false\npromotion_authority=false\n",
  )
}
