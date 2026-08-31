///|
test "reusable residual runtime export round-trips exact startup identity" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let launch = @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y=1,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=512,
  )
  let model_content = "1".repeat(64)
  let model_plan = "2".repeat(64)
  let module_bytes = b"deterministic-residual-rmsnorm-cubin"
  let canonical = export_reusable_fused_residual_runtime(
    model_content_sha256=model_content,
    model_plan_sha256=model_plan,
    target~,
    residual_operation=17,
    norm_operation=18,
    function_symbol="lunaflux_fused_residual_rmsnorm_bf16_v2",
    launch~,
    module_bytes~,
  )
  let source_sha256 = @crypto.bytes_to_hex_string(@crypto.sha256(canonical))
  let admitted = admit_digest_pinned_reusable_fused_residual_runtime(
    canonical, source_sha256,
  )
  assert_eq(admitted.source_digest().as_hex(), source_sha256)
  assert_eq(admitted.model_content_sha256(), model_content)
  assert_eq(admitted.model_plan_sha256(), model_plan)
  assert_eq(admitted.target(), target)
  assert_eq(admitted.operation_indices(), (17, 18))
  assert_eq(
    admitted.function_symbol(),
    "lunaflux_fused_residual_rmsnorm_bf16_v2",
  )
  assert_eq(admitted.launch_dimensions(), launch)
  assert_eq(admitted.module_bytes(), module_bytes)
}

///|
test "reusable residual runtime rejects changed canonical bytes" {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let launch = @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y=1,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=512,
  )
  let canonical = export_reusable_fused_residual_runtime(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    residual_operation=17,
    norm_operation=18,
    function_symbol="lunaflux_fused_residual_rmsnorm_bf16_v2",
    launch~,
    module_bytes=b"deterministic-residual-rmsnorm-cubin",
  )
  try
    ignore(
      admit_digest_pinned_reusable_fused_residual_runtime(
        canonical,
        "f".repeat(64),
      ),
    )
  catch {
    MismatchedFusedArtifact(Manifest) => ()
    _ => fail("unexpected reusable runtime digest rejection")
  } noraise {
    _ => fail("changed reusable runtime digest was accepted")
  }
}
