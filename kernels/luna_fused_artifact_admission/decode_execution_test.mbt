///|
test "closed decode descriptors retain numeric permissions and storage geometry" {
  let laws = [
    "blockwise-f32-probability-v1", "dual-score-blockwise-f32-probability-v2", "blockwise-fma-f32-probability-v3",
    "owned2-blockwise-fma-f32-probability-v4", "owned4-blockwise-f32-probability-v4",
    "owned4-blockwise-fma-f32-probability-v4", "owned8-blockwise-f32-probability-v4",
    "owned8-blockwise-fma-f32-probability-v4", "grouped-head-matrix-bf16-probability-v1",
  ]
  for law in laws {
    for rows in [32, 64] {
      let descriptor = DecodeExecutionDescriptor::from_numeric_law(
        law,
        key_value_tile_rows=rows,
        pipeline_stages=2,
      )
      assert_eq(descriptor.numeric_law(), law)
      assert_eq(descriptor.storage_geometry(), (rows, 2))
      assert_true(
        descriptor
        .function_symbol()
        .has_prefix("lunaflux_attention_decode_tile_compiler_v1_"),
      )
    }
  }
  assert_eq(
    DecodeExecutionDescriptor::from_numeric_law(
      "owned4-blockwise-fma-f32-probability-v4",
      key_value_tile_rows=64,
      pipeline_stages=2,
    ).function_symbol(),
    "lunaflux_attention_decode_tile_compiler_v1_owned4_blockwise_fma_f32_v4",
  )
  assert_eq(
    DecodeExecutionDescriptor::new(
      law=MatrixBf16,
      key_value_tile_rows=64,
      pipeline_stages=1,
    ).function_symbol(),
    "lunaflux_attention_decode_tile_compiler_v1_grouped_head_matrix_bf16_v1",
  )
}

///|
test "extended decode bundle authenticates exact numerical and KV stage identity" {
  let descriptor = DecodeExecutionDescriptor::new(
    law=OwnedContracting(4),
    key_value_tile_rows=64,
    pipeline_stages=2,
  )
  let input = ReusableFusedRuntimeModuleInput::new(
    abi=PagedAttentionReadOnlyExtendedDecodeProductionV11,
    first_operation=7,
    operation_count=1,
    function_symbol=descriptor.function_symbol(),
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=1,
      grid_y=1,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=4096,
    ),
    module_bytes=b"decode-cubin",
    decode_execution=descriptor,
  )
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
  )
  let text = @utf8.decode(bytes)
  assert_true(
    text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v11\n"),
  )
  assert_true(text.contains("decode_key_value_tile_rows=64\n"))
  let runtime = admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  assert_eq(
    runtime.qwen_paged_attention_decode_module().unwrap().decode_execution(),
    Some(descriptor),
  )
}

///|
test "unsupported owner or storage geometry is rejected" {
  let dual = DecodeExecutionDescriptor::new(
    law=DualScoreBlockwise,
    key_value_tile_rows=64,
    pipeline_stages=2,
  )
  assert_true(
    dual.matches_function_symbol(
      "lunaflux_attention_decode_tile_compiler_v1_c460_dual_score_blockwise_f32_v2",
    ),
  )
  assert_false(
    dual.matches_function_symbol(
      "lunaflux_attention_decode_tile_compiler_v1_c0460_dual_score_blockwise_f32_v2",
    ),
  )
  assert_false(
    dual.matches_function_symbol(
      "lunaflux_attention_decode_tile_compiler_v1_c460_owned4_blockwise_f32_v4",
    ),
  )
  let invalid_laws : Array[DecodeNumericLaw] = [
    OwnedSeparate(2),
    OwnedContracting(3),
  ]
  for law in invalid_laws {
    try
      DecodeExecutionDescriptor::new(
        law~,
        key_value_tile_rows=32,
        pipeline_stages=2,
      )
    catch {
      InvalidFusedArtifact(KernelAbi) => ()
      _ => fail("unexpected invalid owner error")
    } noraise {
      _ => fail("invalid score ownership admitted")
    }
  }
  try
    DecodeExecutionDescriptor::new(
      law=DualScoreBlockwise,
      key_value_tile_rows=64,
      pipeline_stages=1,
    )
  catch {
    InvalidFusedArtifact(KernelAbi) => ()
    _ => fail("unexpected invalid storage error")
  } noraise {
    _ => fail("SIMT stage1 admitted")
  }
}
