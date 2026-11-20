///|
test "packaging carries supplied module labels without checking payload checksums" {
  let label = "e".repeat(64)
  let input = @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
    abi=ResidualRmsNormProductionV2,
    first_operation=17,
    operation_count=2,
    function_symbol="lunaflux_residual",
    launch=runtime_bundle_test_launch(128, 1, 512),
    module_bytes=b"opaque-not-checksummed-cubin",
    module_label_sha256=label,
  )
  assert_eq(input.module_digest().as_hex(), label)
  assert_eq(input.function_symbol(), "lunaflux_residual")
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  for _ in 0..<2 {
    let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
      model_content_sha256="1".repeat(64),
      model_plan_sha256="2".repeat(64),
      target~,
      modules=[input],
    )
    let bundle = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
      canonical,
      "f".repeat(64),
    )
    assert_eq(bundle.modules()[0].module_digest().as_hex(), label)
    assert_eq(
      bundle.modules()[0].module_bytes(),
      b"opaque-not-checksummed-cubin",
    )
  }
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
      @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
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
test "reusable bundle rejects unsupported ABI but does not checksum opaque module bytes" {
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
      @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
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
  let labelled = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    hostile_module, hostile_module_digest,
  )
  assert_eq(labelled.modules()[0].module_bytes(), b"\x00")
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
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
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
  let admitted = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
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
      @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
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
      @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
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
