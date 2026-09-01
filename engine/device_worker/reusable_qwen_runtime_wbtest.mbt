///|
fn reusable_qwen_test_plan() -> @plan.ModelPlan raise {
  let model = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=1024,
    hidden_size=128,
    intermediate_size=256,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension=32,
    context_length=256,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  @qwen3.build_paged(
    @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("3".repeat(64)),
    ),
    batch=@qwen3.Qwen3PagedBatchEnvelope::new(4),
  )
}

///|
fn reusable_qwen_test_target(
  compute_minor : Int,
) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor~,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn reusable_qwen_test_module(
  abi : @luna_fused_artifact_admission.ReusableFusedRuntimeKernelAbi,
  first_operation : Int,
  operation_count : Int,
  symbol : String,
) -> @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput raise {
  @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
    abi~,
    first_operation~,
    operation_count~,
    function_symbol=symbol,
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=1,
      grid_y=1,
      grid_z=1,
      block_x=if abi == QwenPagedAttentionReadOnlyProductionV2 {
        32
      } else {
        128
      },
      block_y=1,
      block_z=1,
      shared_memory_bytes=if abi == ResidualRmsNormProductionV2 {
        512
      } else {
        0
      },
    ),
    module_bytes=@utf8.encode(symbol),
  )
}

///|
fn reusable_qwen_test_bundle(
  plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  include_readonly : Bool,
  include_residual? : Bool = false,
) -> @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle raise {
  let modules = [
    reusable_qwen_test_module(
      QwenQkNormPositionedRopePagedKvWriteProductionV1,
      3,
      3,
      "lunaflux_qwen_postprocess",
    ),
  ]
  if include_readonly {
    modules.push(
      reusable_qwen_test_module(
        QwenPagedAttentionReadOnlyProductionV2,
        5,
        1,
        "lunaflux_qwen_readonly",
      ),
    )
  }
  if include_residual {
    modules.push(
      reusable_qwen_test_module(
        ResidualRmsNormProductionV2,
        7,
        2,
        "lunaflux_qwen_residual",
      ),
    )
  }
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256=plan.identity().content().as_hex(),
    model_plan_sha256=plan.identity().plan().as_hex(),
    target~,
    modules~,
  )
  @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
}

///|
test "Qwen reusable runtime validates exact model and target" {
  let plan = reusable_qwen_test_plan()
  let target = reusable_qwen_test_target(0)
  let bundle = reusable_qwen_test_bundle(plan, target, true)
  validate_reusable_qwen_runtime_bundle(plan, target, bundle)
}

///|
test "Qwen reusable runtime rejects model and target mismatch" {
  let plan = reusable_qwen_test_plan()
  let target = reusable_qwen_test_target(0)
  let bundle = reusable_qwen_test_bundle(plan, target, true)
  try
    validate_reusable_qwen_runtime_bundle(
      plan,
      reusable_qwen_test_target(1),
      bundle,
    )
  catch {
    Invalid(DeviceTarget) => ()
    _ => fail("unexpected reusable Qwen target rejection")
  } noraise {
    _ => fail("mismatched reusable Qwen target was accepted")
  }
  let foreign = @qwen3.build_paged(
    @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
      @qwen3_spec.Qwen3ModelSpec::new(
        vocabulary_size=1024,
        hidden_size=128,
        intermediate_size=256,
        layer_count=1,
        attention_head_count=4,
        kv_head_count=2,
        head_dimension=32,
        context_length=256,
        rope_theta=1000000.0,
        weight_dtype=BFloat16,
        kv_dtype=BFloat16,
        tied_embeddings=true,
      ),
      @spec.ContentDigest::from_sha256("4".repeat(64)),
    ),
    batch=@qwen3.Qwen3PagedBatchEnvelope::new(4),
  )
  try validate_reusable_qwen_runtime_bundle(foreign, target, bundle) catch {
    Invalid(ModelIdentity) => ()
    _ => fail("unexpected reusable Qwen model rejection")
  } noraise {
    _ => fail("mismatched reusable Qwen model was accepted")
  }
}

///|
test "Qwen reusable runtime requires ingress and readonly pair" {
  let plan = reusable_qwen_test_plan()
  let target = reusable_qwen_test_target(0)
  let incomplete = reusable_qwen_test_bundle(plan, target, false)
  try validate_reusable_qwen_runtime_bundle(plan, target, incomplete) catch {
    Invalid(ResourceState) => ()
    _ => fail("unexpected incomplete reusable Qwen rejection")
  } noraise {
    _ => fail("incomplete reusable Qwen pair was accepted")
  }
}

///|
test "Qwen v2 residual authority rejects duplicate legacy runtime" {
  let plan = reusable_qwen_test_plan()
  let target = reusable_qwen_test_target(0)
  let bundle = reusable_qwen_test_bundle(
    plan,
    target,
    true,
    include_residual=true,
  )
  validate_reusable_qwen_runtime_bundle(plan, target, bundle)
  let launch = @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=512,
  )
  let legacy_canonical = @luna_fused_artifact_admission.export_reusable_fused_residual_runtime(
    model_content_sha256=plan.identity().content().as_hex(),
    model_plan_sha256=plan.identity().plan().as_hex(),
    target~,
    residual_operation=7,
    norm_operation=8,
    function_symbol="lunaflux_legacy_residual",
    launch~,
    module_bytes=b"legacy-residual",
  )
  let legacy = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_residual_runtime(
    legacy_canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(legacy_canonical)),
  )
  try
    ignore(
      admit_optional_reusable_qwen_runtime_bundle(
        plan,
        target,
        Some(bundle),
        Some(legacy),
        None,
        None,
        None,
        None,
      ),
    )
  catch {
    Invalid(ResourceState) => ()
    _ => fail("unexpected duplicate reusable residual rejection")
  } noraise {
    _ => fail("v2 and legacy residual authority were both accepted")
  }
}
