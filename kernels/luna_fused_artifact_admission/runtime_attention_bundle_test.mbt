///|
test "compiler read-only split ABI roundtrips and cannot coexist with legacy KV-writing split" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let compiler = runtime_bundle_test_input(
    PagedAttentionReadOnlyCompilerDecodeProductionV7,
    5,
    1,
    "compiler_decode",
  )
  let bytes = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[compiler],
  )
  let admitted = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  guard admitted.qwen_paged_attention_decode_split_module() is Some(module_) else {
    fail("compiler module missing")
  }
  assert_eq(
    module_.kernel_abi(),
    PagedAttentionReadOnlyCompilerDecodeProductionV7,
  )
  try
    @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
      model_content_sha256="1".repeat(64),
      model_plan_sha256="2".repeat(64),
      target~,
      modules=[
        compiler,
        runtime_bundle_test_input(
          QwenPagedAttentionReadOnlyDecodeSplitProductionV3,
          5,
          1,
          "legacy_decode",
        ),
      ],
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("conflicting decode contracts accepted")
  }
}

///|
test "reusable fused runtime bundle admits three typed families once" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let inputs = [
    runtime_bundle_test_input(ResidualRmsNormProductionV2, 17, 2, "residual"),
    runtime_bundle_test_input(
      QwenQkNormPositionedRopePagedKvWriteProductionV2,
      3,
      3,
      "qwen_write",
    ),
    runtime_bundle_test_input(
      QwenPagedAttentionReadOnlyProductionV2,
      5,
      1,
      "qwen_attention",
    ),
  ]
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=inputs,
  )
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(canonical))
  let admitted = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical, digest,
  )
  assert_eq(admitted.source_digest().as_hex(), digest)
  assert_eq(admitted.model_content_sha256(), "1".repeat(64))
  assert_eq(admitted.model_plan_sha256(), "2".repeat(64))
  assert_eq(admitted.target(), target)
  assert_eq(admitted.modules().length(), 3)
  guard admitted.module_for(QwenQkNormPositionedRopePagedKvWriteProductionV2)
    is Some(qwen) else {
    fail("Qwen fused write module missing")
  }
  assert_eq(qwen.operation_span(), (3, 3))
  assert_eq(
    admitted.qwen_qknorm_rope_kv_write_operation_indices(),
    Some((3, 4, 5)),
  )
  assert_eq(admitted.qwen_paged_attention_readonly_operation_index(), Some(5))
  assert_eq(admitted.residual_rmsnorm_operation_indices(), Some((17, 18)))
  assert_eq(qwen.function_symbol(), "lunaflux_qwen_write")
  assert_eq(qwen.module_bytes(), b"cubin-qwen_write")
}

///|
test "V4 Qwen full ingress owns projection through KV write" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[
      runtime_bundle_test_input(
        QwenQkvQkNormPositionedRopePagedKvWriteProductionV4,
        2,
        4,
        "qwen_full_ingress",
      ),
      runtime_bundle_test_input(
        QwenPagedAttentionReadOnlyProductionV2,
        5,
        1,
        "qwen_attention",
      ),
    ],
  )
  let admitted = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  guard admitted.qwen_full_attention_ingress_module() is Some(ingress) else {
    fail("Qwen V4 full ingress module missing")
  }
  assert_eq(ingress.operation_span(), (2, 4))
  assert_eq(ingress.function_symbol(), "lunaflux_qwen_full_ingress")
  assert_eq(
    admitted.qwen_qknorm_rope_kv_write_operation_indices(),
    Some((3, 4, 5)),
  )
  assert_eq(admitted.qwen_paged_attention_readonly_operation_index(), Some(5))
  assert_true(admitted.qwen_qknorm_rope_kv_write_module() is None)
}

///|
test "partitioned prefill preserves ordered partial and merge entries" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let attention_operation = 5
  let module_bytes = b"partitioned-prefill-cubin"
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedPartialProductionV4,
        first_operation=attention_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_partial_v1",
        launch=runtime_bundle_test_launch(256, 16, 31952, grid_z=2),
        module_bytes~,
      ),
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedMergeProductionV4,
        first_operation=attention_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_merge_v1",
        launch=runtime_bundle_test_launch(256, 16, 0),
        module_bytes~,
      ),
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitioned8PartialProductionV6,
        first_operation=attention_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_partial_v1_p8",
        launch=runtime_bundle_test_launch(256, 16, 31952, grid_z=8),
        module_bytes~,
      ),
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitioned8MergeProductionV6,
        first_operation=attention_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_merge_v1_p8",
        launch=runtime_bundle_test_launch(256, 16, 0),
        module_bytes~,
      ),
      runtime_bundle_test_input(
        QwenPagedAttentionReadOnlyDecodeSplitProductionV3,
        attention_operation,
        1,
        "qwen_decode",
      ),
    ],
  )
  let admitted = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  guard admitted.qwen_partitioned_prefill_partial_module() is Some(partial) &&
    admitted.qwen_partitioned_prefill_merge_module() is Some(merge) else {
    fail("partitioned prefill module pair missing")
  }
  assert_eq(partial.operation_span(), (attention_operation, 1))
  assert_eq(partial.launch_dimensions().grid_z(), 2)
  assert_eq(merge.operation_span(), (attention_operation, 1))
  assert_eq(merge.launch_dimensions().grid_z(), 1)
  assert_eq(partial.module_digest(), merge.module_digest())
  guard admitted.qwen_partitioned8_prefill_partial_module()
    is Some(deep_partial) &&
    admitted.qwen_partitioned8_prefill_merge_module() is Some(deep_merge) else {
    fail("partitioned8 prefill module pair missing")
  }
  assert_eq(deep_partial.launch_dimensions().grid_z(), 8)
  assert_eq(
    deep_partial.function_symbol(),
    "lunaflux_attention_prefill_partitioned_partial_v1_p8",
  )
  assert_eq(deep_merge.launch_dimensions().grid_z(), 1)
  assert_eq(deep_partial.module_digest(), deep_merge.module_digest())
}

///|
test "V3 Qwen attention module retains readonly and decode split authority" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[
      runtime_bundle_test_input(
        QwenQkNormPositionedRopePagedKvWriteProductionV2,
        3,
        3,
        "qwen_write",
      ),
      runtime_bundle_test_input(
        QwenPagedAttentionReadOnlyDecodeSplitProductionV3,
        5,
        1,
        "qwen_attention_split",
      ),
    ],
  )
  let admitted = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  assert_true(admitted.qwen_paged_attention_readonly_module() is Some(_))
  assert_true(admitted.qwen_paged_attention_decode_split_module() is Some(_))
  assert_eq(admitted.qwen_paged_attention_readonly_operation_index(), Some(5))
}

///|
test "V2 compiler prefill and V3 decode may share one attention operation" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[
      runtime_bundle_test_input(
        QwenQkNormPositionedRopePagedKvWriteProductionV2,
        3,
        3,
        "qwen_write",
      ),
      runtime_bundle_test_input(
        QwenPagedAttentionReadOnlyProductionV2,
        5,
        1,
        "qwen_prefill_compiler",
      ),
      runtime_bundle_test_input(
        QwenPagedAttentionReadOnlyDecodeSplitProductionV3,
        5,
        1,
        "qwen_decode",
      ),
    ],
  )
  let admitted = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  guard admitted.qwen_paged_attention_prefill_module() is Some(prefill) &&
    admitted.qwen_paged_attention_decode_module() is Some(decode) else {
    fail("phase-specific Qwen attention modules missing")
  }
  assert_eq(prefill.function_symbol(), "lunaflux_qwen_prefill_compiler")
  assert_eq(decode.function_symbol(), "lunaflux_qwen_decode")
  assert_eq(admitted.modules().length(), 3)
}

///|
test "prefill frontier roles coexist on one attention operation" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[
      runtime_bundle_test_input(
        QwenPagedAttentionPrefillWideQueryProductionV5,
        5,
        1,
        "qwen_prefill_wide",
      ),
      runtime_bundle_test_input(
        QwenPagedAttentionReadOnlyProductionV2,
        5,
        1,
        "qwen_prefill_selected",
      ),
      runtime_bundle_test_input(
        QwenPagedAttentionReadOnlyDecodeSplitProductionV3,
        5,
        1,
        "qwen_decode",
      ),
    ],
  )
  let admitted = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  guard admitted.qwen_wide_query_prefill_module() is Some(wide) &&
    admitted.qwen_paged_attention_prefill_module() is Some(selected) else {
    fail("prefill frontier modules missing")
  }
  assert_eq(wide.operation_span(), (5, 1))
  assert_eq(wide.function_symbol(), "lunaflux_qwen_prefill_wide")
  assert_eq(selected.function_symbol(), "lunaflux_qwen_prefill_selected")
  assert_eq(admitted.modules().length(), 3)
}
