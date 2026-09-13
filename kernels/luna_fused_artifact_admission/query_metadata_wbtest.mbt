///|
test "query metadata uses an explicit versioned runtime bundle" {
  let input = ReusableFusedRuntimeModuleInput::new(
    abi=QwenPagedAttentionReadOnlyProductionV2,
    first_operation=5,
    operation_count=1,
    function_symbol="attention",
    module_bytes=b"test-module",
    query_tile_rows=64,
    query_metadata_version=1,
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=16,
      grid_y=16,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=49168,
    ),
  )
  let bytes = export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    modules=[input],
  )
  assert_true(@utf8.decode(bytes).contains("bundle.v5"))
  let bundle = admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  assert_eq(
    bundle
    .qwen_paged_attention_readonly_module()
    .unwrap()
    .query_metadata_version(),
    1,
  )
  let downgraded = @utf8.encode(
    @utf8.decode(bytes).replace(old="bundle.v5", new="bundle.v4"),
  )
  try
    admit_digest_pinned_reusable_fused_runtime_bundle(
      downgraded,
      @crypto.bytes_to_hex_string(@crypto.sha256(downgraded)),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("metadata must not be parsed as a legacy row-offset ABI")
  }
}
