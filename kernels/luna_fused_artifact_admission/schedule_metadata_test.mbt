///|
test "runtime bundle round trips semantic tile independently of launch" {
  for tile in [16, 32, 64] {
    let input = ReusableFusedRuntimeModuleInput::new(
      abi=QwenPagedAttentionReadOnlyProductionV2,
      first_operation=5,
      operation_count=1,
      function_symbol="attention",
      launch=@launch_contract.AotLaunchDimensions::new(
        grid_x=2,
        grid_y=16,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=31952,
      ),
      module_bytes=b"module",
      query_tile_rows=tile,
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
    let bundle = admit_digest_pinned_reusable_fused_runtime_bundle(
      bytes,
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    )
    guard bundle.qwen_paged_attention_readonly_module() is Some(module_) else {
      fail("missing attention module")
    }
    assert_eq(module_.query_tile_rows(), Some(tile))
    assert_eq(module_.launch_dimensions().grid_x(), 2)
    let legacy = @utf8.encode(
      @utf8.decode(bytes)
      .replace(old="bundle.v4", new="bundle.v3")
      .replace(old="module_0_query_tile_rows=\{tile}\n", new=""),
    )
    let old_bundle = admit_digest_pinned_reusable_fused_runtime_bundle(
      legacy,
      @crypto.bytes_to_hex_string(@crypto.sha256(legacy)),
    )
    guard old_bundle.qwen_paged_attention_readonly_module() is Some(old_module) else {
      fail("missing legacy module")
    }
    assert_eq(old_module.query_tile_rows(), None)
  }
}
