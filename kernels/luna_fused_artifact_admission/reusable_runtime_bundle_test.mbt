///|
fn runtime_bundle_test_launch(
  block_x : Int,
  grid_y : Int,
  shared_memory_bytes : Int,
  grid_z? : Int = 1,
) -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y~,
    grid_z~,
    block_x~,
    block_y=1,
    block_z=1,
    shared_memory_bytes~,
  )
}

///|
fn runtime_bundle_test_input(
  abi : ReusableFusedRuntimeKernelAbi,
  first_operation : Int,
  operation_count : Int,
  suffix : String,
) -> ReusableFusedRuntimeModuleInput raise FusedArtifactAdmissionError {
  ReusableFusedRuntimeModuleInput::new(
    abi~,
    first_operation~,
    operation_count~,
    function_symbol="lunaflux_" + suffix,
    launch=runtime_bundle_test_launch(
      if abi == QwenPagedAttentionReadOnlyProductionV2 ||
        abi == QwenPagedAttentionReadOnlyDecodeSplitProductionV3 {
        32
      } else {
        128
      },
      if abi == QwenQkNormPositionedRopePagedKvWriteProductionV2 {
        8
      } else {
        1
      },
      if abi == ResidualRmsNormProductionV2 {
        512
      } else {
        0
      },
    ),
    module_bytes=@utf8.encode("cubin-" + suffix),
  )
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
  let canonical = export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=inputs,
  )
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(canonical))
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
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
  let canonical = export_reusable_fused_runtime_bundle(
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
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  guard admitted.qwen_full_attention_ingress_module() is Some(ingress) else {
    fail("Qwen V4 full ingress module missing")
  }
  assert_eq(ingress.operation_span(), (2, 4))
  assert_eq(ingress.function_symbol(), "lunaflux_qwen_full_ingress")
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
  let canonical = export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[
      ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedPartialProductionV4,
        first_operation=attention_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_partial_v1",
        launch=runtime_bundle_test_launch(256, 16, 31952, grid_z=2),
        module_bytes~,
      ),
      ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedMergeProductionV4,
        first_operation=attention_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_merge_v1",
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
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
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
}

///|
test "V3 Qwen attention module retains readonly and decode split authority" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let canonical = export_reusable_fused_runtime_bundle(
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
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
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
  let canonical = export_reusable_fused_runtime_bundle(
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
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
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
test "reusable fused runtime bundle rejects duplicate family input" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let duplicate = [
    runtime_bundle_test_input(ResidualRmsNormProductionV2, 17, 2, "first"),
    runtime_bundle_test_input(ResidualRmsNormProductionV2, 31, 2, "second"),
  ]
  try
    ignore(
      export_reusable_fused_runtime_bundle(
        model_content_sha256="1".repeat(64),
        model_plan_sha256="2".repeat(64),
        target~,
        modules=duplicate,
      ),
    )
  catch {
    MismatchedFusedArtifact(Family) => ()
    _ => fail("unexpected duplicate fused family rejection")
  } noraise {
    _ => fail("duplicate fused family was accepted")
  }
}

///|
test "reusable fused runtime bundle rejects hostile ABI and changed module" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let canonical = export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[
      runtime_bundle_test_input(
        QwenPagedAttentionReadOnlyProductionV2,
        6,
        1,
        "qwen_attention",
      ),
    ],
  )
  let original = @utf8.decode(canonical)
  let hostile_abi = @utf8.encode(
    original.replace(
      old="qwen-paged-attention-readonly-production-v2",
      new="future-unsupported-production-v99",
    ),
  )
  let hostile_abi_digest = @crypto.bytes_to_hex_string(
    @crypto.sha256(hostile_abi),
  )
  try
    ignore(
      admit_digest_pinned_reusable_fused_runtime_bundle(
        hostile_abi, hostile_abi_digest,
      ),
    )
  catch {
    InvalidFusedArtifact(Family) => ()
    _ => fail("unexpected unsupported fused ABI rejection")
  } noraise {
    _ => fail("unsupported fused ABI was accepted")
  }
  let hostile_module = @utf8.encode(
    original.replace(old="637562696e2d7177656e5f617474656e74696f6e", new="00"),
  )
  let hostile_module_digest = @crypto.bytes_to_hex_string(
    @crypto.sha256(hostile_module),
  )
  try
    ignore(
      admit_digest_pinned_reusable_fused_runtime_bundle(
        hostile_module, hostile_module_digest,
      ),
    )
  catch {
    MismatchedFusedArtifact(Module) => ()
    _ => fail("unexpected substituted fused module rejection")
  } noraise {
    _ => fail("substituted fused module was accepted")
  }
}

///|
test "reusable fused runtime ABI enforces family operation cardinality" {
  try
    ignore(
      runtime_bundle_test_input(
        QwenQkNormPositionedRopePagedKvWriteProductionV2,
        2,
        2,
        "truncated_qwen",
      ),
    )
  catch {
    InvalidFusedArtifact(Module) => ()
    _ => fail("unexpected Qwen operation-span rejection")
  } noraise {
    _ => fail("truncated Qwen operation span was accepted")
  }
}

///|
test "Qwen postprocess entry cannot authorize absent readonly attention" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let canonical = export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[
      runtime_bundle_test_input(
        QwenQkNormPositionedRopePagedKvWriteProductionV2,
        3,
        3,
        "qwen_write_only",
      ),
    ],
  )
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  assert_true(admitted.qwen_qknorm_rope_kv_write_module() is Some(_))
  assert_true(admitted.qwen_paged_attention_readonly_module() is None)
  assert_eq(admitted.qwen_paged_attention_readonly_operation_index(), None)
}

///|
test "reusable fused runtime bundle rejects overlapping operation authority" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  try
    ignore(
      export_reusable_fused_runtime_bundle(
        model_content_sha256="1".repeat(64),
        model_plan_sha256="2".repeat(64),
        target~,
        modules=[
          runtime_bundle_test_input(
            ResidualRmsNormProductionV2,
            4,
            2,
            "residual",
          ),
          runtime_bundle_test_input(
            QwenQkNormPositionedRopePagedKvWriteProductionV2,
            3,
            3,
            "qwen_write",
          ),
        ],
      ),
    )
  catch {
    MismatchedFusedArtifact(Binding) => ()
    _ => fail("unexpected overlapping operation-span rejection")
  } noraise {
    _ => fail("overlapping operation authority was accepted")
  }
}

///|
test "Qwen readonly overlap is restricted to the postprocess terminal operation" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  try
    ignore(
      export_reusable_fused_runtime_bundle(
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
            4,
            1,
            "qwen_attention_wrong",
          ),
        ],
      ),
    )
  catch {
    MismatchedFusedArtifact(Binding) => ()
    _ => fail("unexpected nonterminal Qwen attention overlap rejection")
  } noraise {
    _ => fail("nonterminal Qwen attention overlap was accepted")
  }
}
