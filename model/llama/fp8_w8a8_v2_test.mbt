///|
fn fp8_v2_metadata(layer_count? : Int = 1) -> @spec.LlamaModelMetadata raise {
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=32,
      hidden_size=8,
      intermediate_size=16,
      layer_count~,
      attention_head_count=4,
      kv_head_count=2,
      context_length=32,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn fp8_v2_execution(
  policy : @numeric_contract.ActivationScalePolicy,
) -> @numeric_contract.OperationExecutionContract raise {
  @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=policy,
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn fp8_v2_schema_with_execution(
  plan : @plan.ModelPlan,
  index : Int,
  execution : @numeric_contract.OperationExecutionContract,
) -> @numeric_contract.ModelNumericSchema raise {
  let source = plan.numeric_binding().schema()
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  for tensor in source.tensors() {
    tensors.push(tensor)
  }
  let operations : Array[@numeric_contract.NumericOperationSchema] = []
  for position, operation in source.operations() {
    operations.push(
      if position == index {
        @numeric_contract.NumericOperationSchema::new(
          ordinal=@numeric_contract.NumericOperationOrdinal::new(position),
          execution~,
        )
      } else {
        operation
      },
    )
  }
  @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations~,
  )
}

///|
fn fp8_v2_copy_operations(
  source : ReadOnlyArray[@plan.PlanOperation],
) -> Array[@plan.PlanOperation] {
  let result : Array[@plan.PlanOperation] = []
  for operation in source {
    result.push(operation)
  }
  result
}

///|
fn fp8_v2_with_workspace(
  operation : @plan.PlanOperation,
  workspace_bytes : Int64,
) -> @plan.PlanOperation raise {
  let inputs : Array[@plan.ValueRef] = []
  let outputs : Array[@plan.ActivationRef] = []
  for input in operation.inputs() {
    inputs.push(input)
  }
  for output in operation.outputs() {
    outputs.push(output)
  }
  @plan.PlanOperation::new_stateless(
    id=operation.id(),
    kind=operation.kind(),
    inputs~,
    outputs~,
    shape=operation.shape(),
    required_capability=operation.required_capability(),
    workspace_bytes~,
  )
}

///|
fn fp8_v2_rebuild_stateless(
  plan : @plan.ModelPlan,
  operations : ArrayView[@plan.PlanOperation],
  schema : @numeric_contract.ModelNumericSchema,
) -> @plan.ModelPlan raise {
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=plan.identity().content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=plan.shape_constraints(),
    kv_cache_geometry=plan.kv_cache_geometry(),
    workspace_bounds=plan.workspace_bounds(),
    operations~,
    final_output=plan.final_output(),
    kv_execution=StatelessFullContext,
    numeric_schema=schema,
  )
}

///|
test "dense Llama FP8 v2 policy is typed and untied-only" {
  try @llama.LlamaFp8W8A8V2Policy::new(tied_embeddings=true) catch {
    TiedFp8EmbeddingsUnsupported => ()
  } noraise {
    _ => fail("tied FP8 v2 embeddings were admitted")
  }
  assert_eq(
    @llama.LlamaFp8W8A8V2Policy::new(tied_embeddings=false),
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
}

///|
test "dense Llama FP8 v2 binds exact one-stage and compound workspaces" {
  let plan = @llama.build_finite_e4m3_fp8_w8a8_v2(
    fp8_v2_metadata(),
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
  let simple = @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1()
  let compound = @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2()
  let mut simple_count = 0
  let mut gated_count = 0
  for index, operation in plan.operations() {
    let policy = plan.numeric_binding().operations()[index]
      .execution()
      .activation_scale()
    match operation.kind() {
      QkvProjection | OutputProjection | LanguageModelHead => {
        simple_count += 1
        assert_eq(operation.workspace_bytes(), 4L)
        assert_eq(policy, simple)
      }
      GatedMlp => {
        gated_count += 1
        assert_eq(operation.workspace_bytes(), 8L)
        assert_eq(policy, compound)
      }
      _ => {
        assert_eq(operation.workspace_bytes(), 0L)
        assert_eq(
          policy,
          @numeric_contract.ActivationScalePolicy::absent_exact_v1(),
        )
      }
    }
  }
  assert_eq(simple_count, 3)
  assert_eq(gated_count, 1)
}

///|
test "dense Llama FP8 v2 identity is deterministic and leaves v1 unchanged" {
  let metadata = fp8_v2_metadata(layer_count=2)
  let v1 = @llama.build_finite_e4m3_fp8_w8a8_v1(
    metadata,
    @llama.LlamaFp8W8A8Policy::untied_v1(),
  )
  let first = @llama.build_finite_e4m3_fp8_w8a8_v2(
    metadata,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
  let repeated = @llama.build_finite_e4m3_fp8_w8a8_v2(
    metadata,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
  let paged = @llama.build_paged_finite_e4m3_fp8_w8a8_v2(
    metadata,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
  assert_eq(first, repeated)
  assert_true(first.identity().plan() != v1.identity().plan())
  assert_true(first.numeric_binding().digest() != v1.numeric_binding().digest())
  assert_true(first.identity().plan() != paged.identity().plan())
  inspect(
    first.identity().plan().as_hex(),
    content="e035f26c7cf97277a1fb06ced93c5468fca2e92a6d54dd4370178970b3bae15a",
  )
  inspect(
    first.numeric_binding().digest().as_hex(),
    content="74a99f9f32fdf90c1ae0be43516c29ad4e61b304d7deaa100e32c7924f30cfc2",
  )
  for operation in v1.operations() {
    assert_eq(operation.workspace_bytes(), 0L)
  }
  inspect(
    v1.identity().plan().as_hex(),
    content="83393dddd3800f45170b6a9343c1b830c0306e2466c6dfefb153ae3b46a8f49d",
  )
  inspect(
    v1.numeric_binding().digest().as_hex(),
    content="5b595e6132e79caba501c0974ad5002749212324097292d09a66f440160dee48",
  )
}

///|
test "plan validation rejects compound policy substitution across kinds" {
  let plan = @llama.build_finite_e4m3_fp8_w8a8_v2(
    fp8_v2_metadata(),
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
  let operations = fp8_v2_copy_operations(plan.operations())
  let hostile = [
    (
      2,
      fp8_v2_execution(
        @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2(),
      ),
    ),
  ]
  for entry in hostile {
    let (index, execution) = entry
    try
      fp8_v2_rebuild_stateless(
        plan,
        operations,
        fp8_v2_schema_with_execution(plan, index, execution),
      )
    catch {
      @plan.InvalidGraph(NumericOperationBinding) => ()
      _ => fail("unexpected FP8 policy substitution failure")
    } noraise {
      _ => fail("FP8 activation policy was replayed across operation kinds")
    }
  }
}

///|
test "plan validation rejects compound gated workspace substitution" {
  let plan = @llama.build_finite_e4m3_fp8_w8a8_v2(
    fp8_v2_metadata(),
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
  let operations = fp8_v2_copy_operations(plan.operations())
  operations[8] = fp8_v2_with_workspace(operations[8], 4L)
  try
    fp8_v2_rebuild_stateless(plan, operations, plan.numeric_binding().schema())
  catch {
    @plan.InvalidGraph(NumericOperationBinding) => ()
    _ => fail("unexpected compound gated workspace failure")
  } noraise {
    _ => fail("compound gated policy accepted a one-cell workspace")
  }
}

///|
test "plan validation rejects mixed v1 and v2 projection workspaces" {
  let plan = @llama.build_finite_e4m3_fp8_w8a8_v2(
    fp8_v2_metadata(),
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
  for hostile_workspace in [0L, 8L] {
    let operations = fp8_v2_copy_operations(plan.operations())
    operations[2] = fp8_v2_with_workspace(operations[2], hostile_workspace)
    try
      fp8_v2_rebuild_stateless(
        plan,
        operations,
        plan.numeric_binding().schema(),
      )
    catch {
      @plan.InvalidGraph(NumericOperationBinding) => ()
      _ => fail("unexpected mixed FP8 workspace failure")
    } noraise {
      _ => fail("compound FP8 plan admitted a non-v2 simple workspace")
    }
  }
}
