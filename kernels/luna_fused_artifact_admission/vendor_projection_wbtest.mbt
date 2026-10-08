///|
test "vendor projection opt in is canonical and legacy bundles stay disabled" {
  let module_ = ReusableFusedRuntimeModuleInput::new(
    abi=ResidualRmsNormProductionV2,
    first_operation=0,
    operation_count=2,
    function_symbol="lunaflux_residual",
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=32,
      grid_y=1,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=512,
    ),
    module_bytes=b"fixture-not-cubin",
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  for rows in [0, 2, 2048, 1048576] {
    let bytes = export_reusable_fused_runtime_bundle(
      model_content_sha256="1".repeat(64),
      model_plan_sha256="2".repeat(64),
      target~,
      modules=[module_],
      vendor_projection_exact_rows=rows,
    )
    let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
      bytes,
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    )
    assert_eq(admitted.vendor_projection_exact_rows(), rows)
    assert_false(admitted.target().supports_cublas_lt())
  }
  for rows in [-1, 1, 1048577] {
    try
      export_reusable_fused_runtime_bundle(
        model_content_sha256="1".repeat(64),
        model_plan_sha256="2".repeat(64),
        target~,
        modules=[module_],
        vendor_projection_exact_rows=rows,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid exact domain")
    }
  }
}
