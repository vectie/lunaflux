///|
fn reference_domain_admission_test(
  symbol : String,
  coverage : @luna_attention_strategy.AttentionMixedRowCoverage,
  shared_memory_bytes? : Int = 81920,
  accepted? : Bool = true,
) -> Unit raise {
  let plan = qwen_fused_test_plan()
  let base = qwen_fused_test_blueprint(plan)
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=8,
      total_page_count=8,
      max_arena_bytes=100000000L,
    ),
  )
  let blueprint = { ..base, layout, }
  let old = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
  )
  let modules : Array[
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput,
  ] = []
  for module_ in old.modules() {
    if module_.kernel_abi() == QwenPagedAttentionReadOnlyProductionV2 {
      continue
    }
    let (first, count) = module_.operation_span()
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=module_.kernel_abi(),
        first_operation=first,
        operation_count=count,
        function_symbol=module_.function_symbol(),
        launch=module_.launch_dimensions(),
        module_bytes=module_.module_bytes(),
        query_tile_rows?=module_.query_tile_rows(),
      ),
    )
  }
  for
    abi in [
      @luna_fused_artifact_admission.QwenPagedAttentionReadOnlyProductionV2,
      QwenPagedAttentionPrefillWideQueryProductionV5,
    ] {
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi~,
        first_operation=5,
        operation_count=1,
        function_symbol=symbol,
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=blueprint.profile().max_query_rows() +
            (
              blueprint.profile().max_query_tokens() -
              blueprint.profile().max_query_rows() +
              63
            ) /
            64,
          grid_y=16,
          grid_z=1,
          block_x=128,
          block_y=1,
          block_z=1,
          shared_memory_bytes~,
        ),
        module_bytes=b"reference-cubin-fixture",
        query_tile_rows=64,
        prefill_exponential_law=ReferenceBf16ProbabilityBaseTwo,
        allow_approximate_prefill_exponential=true,
      ),
    )
  }
  let bytes = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256=plan.identity().content().as_hex(),
    model_plan_sha256=plan.identity().plan().as_hex(),
    target=blueprint.target(),
    modules~,
  )
  let runtime = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  if !accepted {
    try
      ignore(
        admit_reusable_qwen_fused_attention_spans(
          plan,
          @worker_protocol.ModelPlanGeneration::first(),
          blueprint,
          runtime,
        ),
      )
    catch {
      MismatchedFusedGroupedOperation(KernelAbi) => ()
      _ => fail("wrong reference resource rejection")
    } noraise {
      _ => fail("unsupported reference resource layout admitted")
    }
    return
  }
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_eq(spans.length(), 28)
  assert_eq(spans[0].readonly_query_tile_rows, 64)
  assert_eq(spans[0].mixed_row_coverage, coverage)
  assert_true(spans[0].wide_query_prefill() is None)
  assert_true(spans[0].partitioned_prefill() is None)
  assert_eq(
    spans[0].decode_symbol,
    "lunaflux_attention_decode_tile_compiler_v1",
  )
  try
    ignore(
      admit_reusable_qwen_fused_attention_spans(
        plan,
        @worker_protocol.ModelPlanGeneration::first(),
        base,
        runtime,
      ),
    )
  catch {
    MismatchedFusedGroupedOperation(KernelAbi) => ()
    _ => fail("wrong incompatible page failure")
  } noraise {
    _ => fail("four-row page admitted for reference copy ownership")
  }
}

///|
test "reference AOT prefill admission retains independent decode and deduplicates wide slot" {
  reference_domain_admission_test(
    "lunaflux_attention_prefill_flashattention_bf16_exp2_v1",
    PrefillRowsOnly,
  )
}

///|
test "reference AOT mixed ownership is explicit and pure decode remains independent" {
  reference_domain_admission_test(
    "lunaflux_attention_mixed_flashattention_bf16_exp2_v1",
    AllActiveRows,
  )
}

///|
test "reference prefill schedule reaches worker admission with exact shared geometry" {
  for
    (symbol, coverage) in [
      (
        "lunaflux_attention_prefill_flashattention_bf16_exp2_v1",
        @luna_attention_strategy.PrefillRowsOnly,
      ),
      ("lunaflux_attention_mixed_flashattention_bf16_exp2_v1", AllActiveRows),
    ] {
    reference_domain_admission_test(symbol, coverage, shared_memory_bytes=49152)
    for shared_memory_bytes in [49151, 49153, 32768, 65536] {
      reference_domain_admission_test(
        symbol,
        coverage,
        shared_memory_bytes~,
        accepted=false,
      )
    }
  }
}
