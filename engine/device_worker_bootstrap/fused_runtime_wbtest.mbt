///|
fn bootstrap_reusable_qwen_bundle() -> @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle raise {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let launch = @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    modules=[
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenQkvQkNormPositionedRopePagedKvWriteProductionV4,
        first_operation=2,
        operation_count=4,
        function_symbol="lunaflux_qwen_ingress",
        launch~,
        module_bytes=b"qwen-ingress",
      ),
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionReadOnlyProductionV2,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_qwen_readonly",
        launch~,
        module_bytes=b"qwen-readonly",
      ),
    ],
  )
  @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
}

///|
test "fused runtime schema selects reusable bundle v2 before legacy" {
  match
    classify_fused_runtime_source(
      "schema=lunaflux-reusable-fused-runtime-bundle.v2\nmodule-count=2\n",
    ) {
    ReusableBundleV2 => ()
    _ => fail("reusable fused runtime bundle v2 was not selected")
  }
  match
    classify_fused_runtime_source(
      "schema=lunaflux-reusable-fused-residual-rmsnorm-runtime.v1\n",
    ) {
    ReusableResidualV1 => ()
    _ => fail("legacy reusable residual schema changed")
  }
  match
    classify_fused_runtime_source(
      "schema=lunaflux-fused-production-runtime-source.v1\n",
    ) {
    ProductionV1 => ()
    _ => fail("production fused schema changed")
  }
  match classify_fused_runtime_source("schema=lunaflux-fused-runtime.v1\n") {
    LegacyCatalogV1 => ()
    _ => fail("legacy fused catalog fallback changed")
  }
}

///|
test "Llama and Mistral bootstrap reject reusable Qwen bundle" {
  let loaded : LoadedOptionalFusedRuntime = {
    catalog: None,
    bundle: None,
    production: None,
    reusable_residual: None,
    reusable_bundle: Some(bootstrap_reusable_qwen_bundle()),
  }
  for _ in ["llama", "mistral"] {
    try reject_reusable_qwen_bundle_for_non_qwen(Some(loaded)) catch {
      Invalid(Recipe) => ()
      _ => fail("unexpected non-Qwen reusable runtime rejection")
    } noraise {
      _ => fail("non-Qwen family activated a reusable Qwen runtime bundle")
    }
  }
}
