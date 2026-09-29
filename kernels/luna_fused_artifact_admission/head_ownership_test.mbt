///|
test "head ownership round trips separately from token ownership" {
  for heads in [1, 2, 4] {
    let input = ReusableFusedRuntimeModuleInput::new(
      abi=QwenQkvQkNormPositionedRopePagedKvWriteProductionV4,
      first_operation=2,
      operation_count=4,
      function_symbol="ingress",
      launch=@launch_contract.AotLaunchDimensions::new(
        grid_x=3,
        grid_y=32 / heads,
        grid_z=1,
        block_x=128,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      ),
      module_bytes=b"head-packed-module",
      query_tile_rows=32,
      heads_per_cta=heads,
    )
    for routes in [None, Some(b"opaque-offline-routes")] {
      let bytes = export_reusable_fused_runtime_bundle(
        model_content_sha256="1".repeat(64),
        model_plan_sha256="2".repeat(64),
        target=@catalog.DeviceTarget::new(
          compute_major=12,
          compute_minor=1,
          supports_bf16=true,
          supports_cublas_lt=false,
        ),
        modules=[input],
        attention_bucket_routes?=routes,
      )
      let runtime = admit_digest_pinned_reusable_fused_runtime_bundle(
        bytes,
        @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
      )
      let module_ = runtime.qwen_full_attention_ingress_module().unwrap()
      assert_eq(module_.heads_per_cta(), heads)
      assert_eq(module_.query_tile_rows(), Some(32))
      assert_eq(runtime.attention_bucket_routes(), routes)
      if heads > 1 {
        let text = @utf8.decode(bytes)
        assert_true(
          text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v7\n"),
        )
        for invalid in ["0", "3", "8"] {
          let bad = @utf8.encode(
            text.replace(
              old="heads_per_cta=\{heads}",
              new="heads_per_cta=\{invalid}",
            ),
          )
          try
            admit_digest_pinned_reusable_fused_runtime_bundle(
              bad,
              @crypto.bytes_to_hex_string(@crypto.sha256(bad)),
            )
          catch {
            _ => ()
          } noraise {
            _ => fail("invalid head ownership accepted")
          }
        }
      }
    }
  }
}

///|
test "head packing cannot be attached to a different semantic family or missing row metadata" {
  let cases : Array[(ReusableFusedRuntimeKernelAbi, Int, Int?)] = [
    (ResidualRmsNormProductionV2, 2, Some(32)),
    (QwenQkNormPositionedRopePagedKvWriteProductionV2, 3, Some(32)),
    (QwenQkvQkNormPositionedRopePagedKvWriteProductionV4, 4, None),
  ]
  for (abi, count, rows) in cases {
    try
      ReusableFusedRuntimeModuleInput::new(
        abi~,
        first_operation=2,
        operation_count=count,
        function_symbol="invalid",
        module_bytes=b"bytes",
        query_tile_rows?=rows,
        heads_per_cta=2,
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=1,
          grid_y=16,
          grid_z=1,
          block_x=128,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid packing family accepted")
    }
  }
}
