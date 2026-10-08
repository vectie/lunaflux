///|
fn reference_numeric_test(
  symbol : String,
  coverage : @luna_attention_strategy.AttentionMixedRowCoverage,
) -> Unit raise {
  for law in [StrictNaturalExponential, ApproximateBaseTwoExponential] {
    expect_prefill_numeric_rejection(() => {
      reusable_runtime_validate_prefill_exponential(
        QwenPagedAttentionReadOnlyProductionV2,
        law,
        symbol,
        true,
      )
    })
  }
  expect_prefill_numeric_rejection(() => {
    reusable_runtime_validate_prefill_exponential(
      QwenPagedAttentionReadOnlyProductionV2,
      ReferenceBf16ProbabilityBaseTwo,
      symbol,
      false,
    )
  })
  expect_prefill_numeric_rejection(() => {
    reusable_runtime_validate_prefill_exponential(
      QwenPagedAttentionPrefillPartitionedPartialProductionV4,
      ReferenceBf16ProbabilityBaseTwo,
      symbol,
      true,
    )
  })
  let module_ = ReusableFusedRuntimeModuleInput::new(
    abi=QwenPagedAttentionReadOnlyProductionV2,
    first_operation=5,
    operation_count=1,
    function_symbol=symbol,
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=64,
      grid_y=16,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=81920,
    ),
    query_tile_rows=64,
    module_bytes=b"reference-prefill-fixture-not-a-cubin",
    prefill_exponential_law=ReferenceBf16ProbabilityBaseTwo,
    allow_approximate_prefill_exponential=true,
  )
  let bytes = export_reusable_fused_runtime_bundle(
    model_content_sha256="a".repeat(64),
    model_plan_sha256="b".repeat(64),
    target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    modules=[module_],
  )
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  assert_eq(
    admitted.modules()[0].prefill_exponential_law(),
    ReferenceBf16ProbabilityBaseTwo,
  )
  assert_eq(admitted.modules()[0].mixed_row_coverage(), coverage)
  let changed = @utf8.encode(
    @utf8.decode(bytes).replace(
      old="reference-flashattention-bf16-probability-exp2-v1",
      new="approx-base2-f32-v1",
    ),
  )
  expect_prefill_numeric_rejection(() => {
    ignore(
      admit_digest_pinned_reusable_fused_runtime_bundle(
        changed,
        @crypto.bytes_to_hex_string(@crypto.sha256(changed)),
      ),
    )
  })
}

///|
test "reference prefill has an explicit numerical law, never an expf alias" {
  reference_numeric_test(
    "lunaflux_attention_prefill_flashattention_bf16_exp2_v1",
    PrefillRowsOnly,
  )
}

///|
test "reference mixed domain survives canonical serialization with its distinct symbol" {
  reference_numeric_test(
    "lunaflux_attention_mixed_flashattention_bf16_exp2_v1",
    AllActiveRows,
  )
}
