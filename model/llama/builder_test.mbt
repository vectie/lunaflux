///|
fn llama_test_metadata() -> @spec.LlamaModelMetadata raise {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=2,
    context_length=64,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let content = @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  @spec.LlamaModelMetadata::from_verified_content(spec, content)
}

///|
fn copy_operations(
  source : ReadOnlyArray[@plan.PlanOperation],
) -> Array[@plan.PlanOperation] {
  let output : Array[@plan.PlanOperation] = []
  for operation in source {
    output.push(operation)
  }
  output
}

///|
fn replace_attention_layer(
  operation : @plan.PlanOperation,
  layer : @plan.KvLayerId,
) -> @plan.PlanOperation raise {
  let inputs : Array[@plan.ValueRef] = []
  let outputs : Array[@plan.ActivationRef] = []
  let runtime_inputs : Array[@plan.OperationRuntimeInput] = []
  for input in operation.inputs() {
    inputs.push(input)
  }
  for output in operation.outputs() {
    outputs.push(output)
  }
  for input in operation.runtime_inputs() {
    runtime_inputs.push(input)
  }
  @plan.PlanOperation::new_with_execution(
    id=operation.id(),
    kind=operation.kind(),
    inputs~,
    outputs~,
    shape=operation.shape(),
    required_capability=operation.required_capability(),
    workspace_bytes=operation.workspace_bytes(),
    kv_layer=layer,
    runtime_inputs~,
    persistent_state=KvCacheReadWrite(layer),
  )
}

///|
fn replace_attention_runtime_inputs(
  operation : @plan.PlanOperation,
  runtime_inputs : ArrayView[@plan.OperationRuntimeInput],
) -> @plan.PlanOperation raise {
  let inputs : Array[@plan.ValueRef] = []
  let outputs : Array[@plan.ActivationRef] = []
  for input in operation.inputs() {
    inputs.push(input)
  }
  for output in operation.outputs() {
    outputs.push(output)
  }
  let layer = operation.kv_layer().unwrap()
  @plan.PlanOperation::new_with_execution(
    id=operation.id(),
    kind=operation.kind(),
    inputs~,
    outputs~,
    shape=operation.shape(),
    required_capability=operation.required_capability(),
    workspace_bytes=operation.workspace_bytes(),
    kv_layer=layer,
    runtime_inputs~,
    persistent_state=KvCacheReadWrite(layer),
  )
}

///|
test "dense Llama builder emits complete operations in semantic order" {
  let model_plan = @llama.build(llama_test_metadata())
  let operations = model_plan.operations()
  assert_eq(operations.length(), 21)
  let expected : Array[@plan.OperationKind] = [
    Embedding,
    RmsNorm,
    QkvProjection,
    RotaryEmbedding,
    CausalAttention,
    OutputProjection,
    ResidualAdd,
    RmsNorm,
    GatedMlp,
    ResidualAdd,
    RmsNorm,
    QkvProjection,
    RotaryEmbedding,
    CausalAttention,
    OutputProjection,
    ResidualAdd,
    RmsNorm,
    GatedMlp,
    ResidualAdd,
    RmsNorm,
    LanguageModelHead,
  ]
  for index, operation in operations {
    assert_eq(operation.id().as_int(), index)
    assert_eq(operation.kind(), expected[index])
    assert_eq(operation.outputs()[0].as_int(), index)
  }
}

///|
test "dense Llama dimensions and workspace are explicit and bounded" {
  let model_plan = @llama.build(llama_test_metadata())
  let constraints = model_plan.shape_constraints()
  assert_eq(constraints.max_batch_rows(), 1)
  assert_eq(constraints.max_sequence_tokens(), 64)
  assert_eq(constraints.vocabulary_size(), 128)
  assert_eq(constraints.hidden_size(), 16)
  assert_eq(constraints.intermediate_size(), 32)
  assert_eq(constraints.attention_head_count(), 4)
  assert_eq(constraints.kv_head_count(), 2)
  assert_eq(constraints.head_dimension(), 4)
  let geometry = model_plan.kv_cache_geometry()
  assert_eq(geometry.layer_count(), 2)
  assert_eq(geometry.kv_head_count(), 2)
  assert_eq(geometry.head_dimension(), 4)
  assert_eq(geometry.dtype(), BFloat16)
  assert_eq(geometry.layout(), LayerMajorSplitKeyValueV1)
  assert_eq(geometry.layout().stable_version(), 1)
  assert_eq(model_plan.workspace_bounds().max_temporary_bytes(), 8192L)
  assert_eq(model_plan.kv_execution(), StatelessFullContext)
  for layer in 0..<2 {
    let rotary = model_plan.operations()[3 + layer * 9]
    let attention = model_plan.operations()[4 + layer * 9]
    assert_eq(rotary.kv_layer(), Some(@plan.KvLayerId::new(layer)))
    assert_eq(attention.kv_layer(), Some(@plan.KvLayerId::new(layer)))
    assert_eq(rotary.runtime_inputs().length(), 0)
    assert_eq(attention.runtime_inputs().length(), 0)
    assert_eq(attention.persistent_state(), NoPersistentState)
  }
  guard model_plan.operations()[2].shape()
    is ProjectionShape(input_width=16, output_width=32) else {
    fail("unexpected QKV projection dimensions")
  }
  guard model_plan.operations()[1].shape()
    is NormalizationShape(width=16, epsilon=0.000001) else {
    fail("unexpected RMS normalization semantics")
  }
  guard model_plan.operations()[3].shape()
    is RotaryShape(
      query_heads=4,
      kv_heads=2,
      head_dimension=4,
      rope_theta=10000.0
    ) else {
    fail("unexpected rotary dimensions or parameters")
  }
  guard model_plan.operations()[20].shape()
    is ProjectionShape(input_width=16, output_width=128) else {
    fail("unexpected language-model head dimensions")
  }
}

///|
test "paged Llama plan owns exact layered runtime and persistent-state semantics" {
  let metadata = llama_test_metadata()
  let full_context = @llama.build(metadata)
  let paged = @llama.build_paged(metadata)
  assert_eq(paged.kv_execution(), PagedKeyValue)
  assert_true(paged.identity() != full_context.identity())
  assert_eq(paged.identity().content(), full_context.identity().content())
  assert_eq(
    paged.identity().plan(),
    metadata.spec().canonical_paged_plan_digest(),
  )
  let expected_attention_inputs : Array[@plan.OperationRuntimeInput] = [
    StepCounts,
    QueryPositions,
    QueryRowOffsets,
    SequenceLengths,
    PageTableOffsets,
    PageTablePageIndices,
  ]
  for layer in 0..<2 {
    let layer_id = @plan.KvLayerId::new(layer)
    let rotary = paged.operations()[3 + layer * 9]
    let attention = paged.operations()[4 + layer * 9]
    assert_eq(rotary.kv_layer(), Some(layer_id))
    assert_eq(rotary.runtime_inputs().length(), 2)
    assert_eq(rotary.runtime_inputs()[0], StepCounts)
    assert_eq(rotary.runtime_inputs()[1], QueryPositions)
    assert_eq(rotary.persistent_state(), NoPersistentState)
    assert_eq(
      rotary.required_capability(),
      @plan.KernelCapabilityId::positioned_rotary_embedding(),
    )
    assert_eq(attention.kv_layer(), Some(layer_id))
    assert_eq(
      attention.runtime_inputs().length(),
      expected_attention_inputs.length(),
    )
    for index, input in attention.runtime_inputs() {
      assert_eq(input, expected_attention_inputs[index])
    }
    assert_eq(attention.persistent_state(), KvCacheReadWrite(layer_id))
    assert_eq(
      attention.required_capability(),
      @plan.KernelCapabilityId::paged_causal_attention(),
    )
  }
}

///|
test "plan validation rejects inferred, partial, and mode-mismatched KV ownership" {
  let paged = @llama.build_paged(llama_test_metadata())
  let wrong_layer = copy_operations(paged.operations())
  wrong_layer[13] = replace_attention_layer(
    wrong_layer[13],
    @plan.KvLayerId::new(0),
  )
  try
    @plan.ModelPlan::new_with_kv_execution(
      identity=paged.identity(),
      shape_constraints=paged.shape_constraints(),
      kv_cache_geometry=paged.kv_cache_geometry(),
      workspace_bounds=paged.workspace_bounds(),
      operations=wrong_layer,
      final_output=paged.final_output(),
      kv_execution=PagedKeyValue,
    )
  catch {
    InvalidGraph(KvLayerOrder) => ()
    _ => fail("unexpected KV layer-order failure")
  } noraise {
    _ => fail("expected wrong KV layer rejection")
  }
  let full_context = @llama.build(llama_test_metadata())
  let incomplete : Array[@plan.PlanOperation] = []
  for index in 0..<10 {
    incomplete.push(full_context.operations()[index])
  }
  try
    @plan.ModelPlan::new_with_kv_execution(
      identity=full_context.identity(),
      shape_constraints=full_context.shape_constraints(),
      kv_cache_geometry=full_context.kv_cache_geometry(),
      workspace_bounds=full_context.workspace_bounds(),
      operations=incomplete,
      final_output=@plan.ActivationRef::new(9),
      kv_execution=StatelessFullContext,
    )
  catch {
    InvalidGraph(KvLayerCoverage) => ()
    _ => fail("unexpected KV coverage failure")
  } noraise {
    _ => fail("expected incomplete KV coverage rejection")
  }
  try
    @plan.ModelPlan::new_with_kv_execution(
      identity=paged.identity(),
      shape_constraints=paged.shape_constraints(),
      kv_cache_geometry=paged.kv_cache_geometry(),
      workspace_bounds=paged.workspace_bounds(),
      operations=copy_operations(paged.operations()),
      final_output=paged.final_output(),
      kv_execution=StatelessFullContext,
    )
  catch {
    InvalidGraph(KvExecutionModeMismatch) => ()
    _ => fail("unexpected KV execution-mode failure")
  } noraise {
    _ => fail("expected execution-mode rejection")
  }
  let incomplete_runtime = copy_operations(paged.operations())
  incomplete_runtime[4] = replace_attention_runtime_inputs(
    incomplete_runtime[4],
    [
      StepCounts,
      QueryPositions,
      QueryRowOffsets,
      SequenceLengths,
      PageTableOffsets,
    ],
  )
  try
    @plan.ModelPlan::new_with_kv_execution(
      identity=paged.identity(),
      shape_constraints=paged.shape_constraints(),
      kv_cache_geometry=paged.kv_cache_geometry(),
      workspace_bounds=paged.workspace_bounds(),
      operations=incomplete_runtime,
      final_output=paged.final_output(),
      kv_execution=PagedKeyValue,
    )
  catch {
    InvalidGraph(KvExecutionContract) => ()
    _ => fail("unexpected paged runtime-input failure")
  } noraise {
    _ => fail("expected incomplete paged runtime-input rejection")
  }
}

///|
test "dense Llama plan construction is deterministic" {
  let first = @llama.build(llama_test_metadata())
  let second = @llama.build(llama_test_metadata())
  assert_eq(first, second)
  assert_eq(first.identity(), llama_test_metadata().identity())
}

///|
test "required semantic capabilities are deduplicated in first-use order" {
  let capabilities = @llama.build(llama_test_metadata()).required_capabilities()
  let expected = [
    @plan.KernelCapabilityId::embedding_lookup(),
    @plan.KernelCapabilityId::rms_norm(),
    @plan.KernelCapabilityId::qkv_projection(),
    @plan.KernelCapabilityId::rotary_embedding(),
    @plan.KernelCapabilityId::causal_attention(),
    @plan.KernelCapabilityId::dense_projection(),
    @plan.KernelCapabilityId::residual_add(),
    @plan.KernelCapabilityId::gated_mlp(),
  ]
  assert_eq(capabilities.length(), expected.length())
  for index, capability in capabilities {
    assert_eq(capability, expected[index])
    assert_eq(capability.as_int(), index + 1)
  }
}
