///|
fn device_sampling_test_plan() -> @luna_sampling_strategy.SamplingDevicePlan {
  try! @luna_sampling_strategy.plan_device_sampling(
    maximum_rows=4,
    vocabulary_size=151936,
    maximum_top_k=64,
    maximum_workspace_bytes=3072L,
  )
}

///|
fn device_sampling_test_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn device_sampling_test_compiler() -> @luna_specialization.LunaCompilerPolicy {
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="a".repeat(64),
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=96,
  )
}

///|
fn device_sampling_test_limits() -> DeviceSamplingAotLimits {
  try! DeviceSamplingAotLimits::new(
    max_source_bytes=32768,
    max_recipe_bytes=4096,
    max_module_bytes=65536L,
  )
}

///|
test "restrictive top-k lowering fixes ABI order RNG and nucleus semantics" {
  let first = @luna_cuda_sampling_aot.lower_device_sampling_cuda_aot(
    device_sampling_test_plan(),
    device_sampling_test_target(),
    device_sampling_test_compiler(),
    device_sampling_test_limits(),
  )
  let second = @luna_cuda_sampling_aot.lower_device_sampling_cuda_aot(
    device_sampling_test_plan(),
    device_sampling_test_target(),
    device_sampling_test_compiler(),
    device_sampling_test_limits(),
  )
  assert_eq(first.source_bytes(), second.source_bytes())
  assert_eq(first.source_sha256(), second.source_sha256())
  assert_eq(first.recipe_sha256(), second.recipe_sha256())
  let source = @utf8.decode(first.source_bytes())
  assert_true(source.contains("sizeof(LunaSamplingParametersV2) == 40"))
  assert_true(source.contains("const int32_t *counts"))
  assert_true(source.contains("const int32_t *row_offsets"))
  assert_true(source.contains("const __nv_bfloat16 *logits"))
  assert_true(source.contains("const LunaSamplingParametersV2 *parameters"))
  assert_true(source.contains("void *workspace"))
  assert_true(
    source.contains("value == previous_value && token > previous_token"),
  )
  assert_true(source.contains("0x9e3779b97f4a7c15ULL"))
  assert_true(source.contains("cumulative < target"))
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_true(recipe.contains("maximum_top_k=64"))
  assert_true(recipe.contains("workspace_bytes=3072"))
  assert_true(recipe.contains("manifest_bindable=false"))
  let abi = @luna_cuda_sampling_aot.device_sampling_cuda_abi_v2(first.plan())
  assert_eq(abi.operands(), [
    SamplingCounts,
    SamplingRowOffsets,
    SamplingBf16Logits,
    SamplingParametersV2,
    SamplingWorkspace,
    SamplingResults,
  ])
  assert_eq(abi.grid_x(), 4)
  assert_eq(abi.block_x(), 256)
  assert_eq(abi.parameter_bytes(), 160L)
  assert_eq(abi.workspace_bytes(), 3072L)
  assert_eq(abi.result_bytes(), 32L)
  assert_true(abi.manifest_bindable())
}

///|
test "stochastic artifact requires deterministic offline builds" {
  let lowering = @luna_cuda_sampling_aot.lower_device_sampling_cuda_aot(
    device_sampling_test_plan(),
    device_sampling_test_target(),
    device_sampling_test_compiler(),
    device_sampling_test_limits(),
  )
  let cubin = b"deterministic-device-sampling-cubin"
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(cubin))
  let receipt = DeviceSamplingCompileReceipt::from_offline_builder(
    source_sha256=lowering.source_sha256(),
    recipe_sha256=lowering.recipe_sha256(),
    toolchain_sha256="a".repeat(64),
    driver_sha256="b".repeat(64),
    first_module_sha256=digest,
    second_module_sha256=digest,
  )
  let artifact = @luna_cuda_sampling_aot.admit_device_sampling_cuda_artifact(
    lowering,
    cubin,
    cubin,
    receipt,
    device_sampling_test_limits(),
  )
  assert_eq(
    artifact.function_symbol(),
    "lunaflux_sample_restrictive_topk_bf16_rows_v2",
  )
  assert_eq(artifact.plan().maximum_top_k(), 64)
  assert_false(artifact.manifest_bindable())
}
