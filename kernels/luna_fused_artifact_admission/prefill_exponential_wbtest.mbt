///|
fn prefill_numeric_test_input(
  law : PrefillExponentialLaw,
  allow : Bool,
  abi? : ReusableFusedRuntimeKernelAbi = QwenPagedAttentionReadOnlyProductionV2,
) -> ReusableFusedRuntimeModuleInput raise FusedArtifactAdmissionError {
  ReusableFusedRuntimeModuleInput::new(
    abi~,
    first_operation=5,
    operation_count=1,
    function_symbol=if law == ApproximateBaseTwoExponential {
      "lunaflux_attention_prefill_tile_compiler_exp2_v1"
    } else {
      "lunaflux_attention_prefill_tile_compiler_v1"
    },
    launch=try! @launch_contract.AotLaunchDimensions::new(
      grid_x=32,
      grid_y=16,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=49168,
    ),
    query_tile_rows=64,
    module_bytes=b"qualified-prefill-fixture",
    prefill_exponential_law=law,
    allow_approximate_prefill_exponential=allow,
  )
}

///|
fn expect_prefill_numeric_rejection(body : () -> Unit raise) -> Unit raise {
  try body() catch {
    FusedArtifactAdmissionError::InvalidFusedArtifact(KernelAbi) => ()
    _ => fail("unexpected numeric-law rejection")
  } noraise {
    _ => fail("missing explicit numerical rejection")
  }
}

///|
test "approximate prefill law binds distinct startup permission and module identity" {
  expect_prefill_numeric_rejection(() => {
    ignore(prefill_numeric_test_input(ApproximateBaseTwoExponential, false))
  })
  expect_prefill_numeric_rejection(() => {
    ignore(
      prefill_numeric_test_input(
        ApproximateBaseTwoExponential,
        true,
        abi=PagedAttentionReadOnlyCompilerDecodeProductionV7,
      ),
    )
  })
  let module_ = prefill_numeric_test_input(ApproximateBaseTwoExponential, true)
  let bytes = export_reusable_fused_runtime_bundle(
    model_content_sha256="a".repeat(64),
    model_plan_sha256="b".repeat(64),
    target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    modules=[module_],
  )
  let text = @utf8.decode(bytes)
  assert_true(
    text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v12\n"),
  )
  assert_true(text.contains("allow_approximate_prefill_exponential=1\n"))
  assert_true(
    text.contains("module_0_prefill_exponential_law=approx-base2-f32-v1\n"),
  )
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  assert_eq(
    admitted.modules()[0].prefill_exponential_law(),
    ApproximateBaseTwoExponential,
  )
  for
    (old, new_) in [
      (
        "allow_approximate_prefill_exponential=1", "allow_approximate_prefill_exponential=0",
      ),
      (
        "prefill_exponential_law=approx-base2-f32-v1", "prefill_exponential_law=strict-natural-exponential-v1",
      ),
    ] {
    let tampered = @utf8.encode(text.replace(old~, new=new_))
    expect_prefill_numeric_rejection(() => {
      ignore(
        admit_digest_pinned_reusable_fused_runtime_bundle(
          tampered,
          @crypto.bytes_to_hex_string(@crypto.sha256(tampered)),
        ),
      )
    })
  }
}

///|
test "legacy strict bundle cannot admit an exp2 symbol without numerical metadata" {
  let bytes = export_reusable_fused_runtime_bundle(
    model_content_sha256="a".repeat(64),
    model_plan_sha256="b".repeat(64),
    target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    modules=[prefill_numeric_test_input(StrictNaturalExponential, false)],
  )
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  assert_eq(
    admitted.modules()[0].prefill_exponential_law(),
    StrictNaturalExponential,
  )
  let tampered = @utf8.encode(
    @utf8.decode(bytes).replace(
      old="lunaflux_attention_prefill_tile_compiler_v1",
      new="lunaflux_attention_prefill_tile_compiler_exp2_v1",
    ),
  )
  expect_prefill_numeric_rejection(() => {
    ignore(
      admit_digest_pinned_reusable_fused_runtime_bundle(
        tampered,
        @crypto.bytes_to_hex_string(@crypto.sha256(tampered)),
      ),
    )
  })
}
