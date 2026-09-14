///|
fn bootstrap_reusable_qwen_bundle(
  query_metadata? : Bool = false,
  attention_bucket_routes? : Bytes,
) -> @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle raise {
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
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target~,
    attention_bucket_routes?,
    modules=[
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenQkNormPositionedRopePagedKvWriteProductionV2,
        first_operation=3,
        operation_count=3,
        function_symbol="lunaflux_qwen_postprocess",
        launch~,
        module_bytes=b"qwen-ingress",
      ),
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionReadOnlyProductionV2,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_qwen_readonly",
        launch~,
        query_tile_rows?=if query_metadata { Some(64) } else { None },
        query_metadata_version=if query_metadata { 1 } else { 0 },
        module_bytes=b"qwen-readonly",
      ),
    ],
  )
  match classify_fused_runtime_source(@utf8.decode(canonical)) {
    ReusableBundle => ()
    _ =>
      fail("exported bundle must select the reusable-bundle bootstrap parser")
  }
  @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
}

///|
test "fused runtime schema selects reusable bundles including measured routes v6" {
  for version in [3, 4, 5, 6] {
    match
      classify_fused_runtime_source(
        "schema=lunaflux-reusable-fused-runtime-bundle.v\{version}\nmodule-count=2\n",
      ) {
      ReusableBundle => ()
      _ => fail("reusable fused runtime bundle was not selected")
    }
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
test "exported metadata bundle reaches the worker bootstrap bundle parser" {
  ignore(bootstrap_reusable_qwen_bundle(query_metadata=true))
}

///|
test "exported measured route bundle reaches bootstrap without losing observations" {
  let routes = b"luna-attention-buckets-v1\nscope\tfixture\nrecord\tprefill\t1\t64\t64\t1\t100\t3\n"
  let bundle = bootstrap_reusable_qwen_bundle(
    query_metadata=true,
    attention_bucket_routes=routes,
  )
  assert_eq(bundle.attention_bucket_routes(), Some(routes))
}

///|
test "snapshot envelope includes hex route table and rejects arithmetic overflow" {
  assert_eq(fused_runtime_snapshot_limit(1L), 2162692L)
  let largest = (0x7fffffffffffffffL - 2162688L) / 4L
  assert_true(fused_runtime_snapshot_limit(largest) > 0L)
  for invalid in [0L, -1L, largest + 1L] {
    try fused_runtime_snapshot_limit(invalid) catch {
      Invalid(RecipeLimit) => ()
      _ => fail("wrong error")
    } noraise {
      _ => fail("invalid snapshot limit accepted")
    }
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
