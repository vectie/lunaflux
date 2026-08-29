///|
fn paged_kv_write_bool(value : Bool) -> String {
  if value {
    "true"
  } else {
    "false"
  }
}

///|
fn paged_kv_write_operand_record(
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
fn paged_kv_write_recipe(
  plan : @plan.ModelPlan,
  attention : @plan.PlanOperation,
  input : PositionedQkvActivation,
  layer : @plan.KvLayerId,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  symbol : String,
  dispatch_canary : UInt,
  source : PagedKvWriteDigest,
  fallback : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-positioned-paged-kv-write-candidate.v1\n" +
    "qualification_only=true\nmanifest_bindable=false\npromotion_authority=false\n" +
    "bounds_authority=runtime-validated-step-descriptor\n" +
    "model_content_sha256=" +
    plan.identity().content().as_hex() +
    "\n" +
    "model_plan_sha256=" +
    plan.identity().plan().as_hex() +
    "\n" +
    "attention_operation=" +
    attention.id().as_int().to_string() +
    "\n" +
    "positioned_qkv_activation=" +
    input.qkv.as_int().to_string() +
    "\n" +
    "kv_layer=" +
    layer.as_int().to_string() +
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
    paged_kv_write_bool(compiler.fused_multiply_add()) +
    "\n" +
    "reassociate=" +
    paged_kv_write_bool(compiler.reassociate()) +
    "\n" +
    "max_registers=" +
    compiler.max_registers().to_string() +
    "\n" +
    "profile=" +
    profile.max_query_rows().to_string() +
    "," +
    profile.max_query_tokens().to_string() +
    "," +
    profile.max_page_table_entries().to_string() +
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
    "symbol=" +
    symbol +
    "\ndispatch_canary=" +
    dispatch_canary.to_string() +
    "\nabi=" +
    PagedKvWriteKernelAbi::PositionedQkvPagedWriteBf16V1.canonical_name() +
    "\n" +
    "grid=" +
    dimensions.grid_x().to_string() +
    ",1,1\nblock=256,1,1\n" +
    "operands=" +
    paged_kv_write_operand_record(operands) +
    "\n" +
    "source_sha256=" +
    source.as_hex() +
    "\n" +
    "fallback_family=monolithic-paged-write-and-attend\n" +
    "fallback_operation=" +
    fallback.operation_id().as_int().to_string() +
    "\n" +
    "fallback_source_sha256=" +
    fallback.source_digest().as_hex() +
    "\n" +
    "fallback_recipe_sha256=" +
    fallback.recipe_digest().as_hex() +
    "\n",
  )
}
