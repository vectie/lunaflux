///|
fn producer_fixture_plan() -> @plan.ModelPlan raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=16,
    hidden_size=4,
    intermediate_size=8,
    layer_count=2,
    attention_head_count=2,
    kv_head_count=1,
    context_length=4,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("a".repeat(64)),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::new(2),
  )
}

///|
fn producer_fixture_layout() -> @device_layout.DeviceKvLayout raise {
  let plan = producer_fixture_plan()
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=2,
      total_page_count=8,
      max_arena_bytes=8192L,
    ),
  )
}

///|
fn producer_fixture_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn producer_fixture_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=4,
  )
}

///|
fn producer_fixture_compiler() -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="3".repeat(64),
      major=13,
      minor=1,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn fixture_runtime_bytes(
  input : @plan.OperationRuntimeInput,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  match input {
    StepCounts => 20L
    QueryPositions => profile.max_query_tokens().to_int64() * 4L
    QueryRowOffsets | PageTableOffsets =>
      (profile.max_query_rows().to_int64() + 1L) * 4L
    SequenceLengths => profile.max_query_rows().to_int64() * 4L
    PageTablePageIndices => profile.max_page_table_entries().to_int64() * 4L
  }
}

///|
fn fixture_operation_width(operation : @plan.PlanOperation) -> Int {
  match operation.shape() {
    EmbeddingShape(width~, ..) => width
    NormalizationShape(width~, ..) => width
    QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, ..) =>
      (query_heads + 2 * kv_heads) * head_dimension
    ProjectionShape(output_width~, ..) => output_width
    RotaryShape(query_heads~, kv_heads~, head_dimension~, ..) =>
      (query_heads + 2 * kv_heads) * head_dimension
    AttentionShape(query_heads~, head_dimension~, ..) =>
      query_heads * head_dimension
    ElementwiseShape(width~) => width
    GatedMlpShape(hidden_size~, ..) => hidden_size
  }
}

///|
fn fixture_activation_bytes(
  plan : @plan.ModelPlan,
  reference : @plan.ActivationRef,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  fixture_operation_width(plan.operations()[reference.as_int()]).to_int64() *
  profile.max_query_tokens().to_int64() *
  2L
}

///|
fn fixture_weight_bytes(
  operation : @plan.PlanOperation,
  weight_index : Int,
) -> Int64 {
  match operation.shape() {
    EmbeddingShape(vocabulary_size~, width~) =>
      vocabulary_size.to_int64() * width.to_int64() * 2L
    NormalizationShape(width~, ..) => width.to_int64() * 2L
    QkNormalizationShape(head_dimension~, ..) =>
      if weight_index < 2 {
        head_dimension.to_int64() * 2L
      } else {
        abort("Q/K norm fixture has only two weights")
      }
    ProjectionShape(input_width~, output_width~) =>
      if operation.kind() == QkvProjection {
        let kv_width = (output_width - input_width) / 2
        let rows = if weight_index == 0 { input_width } else { kv_width }
        rows.to_int64() * input_width.to_int64() * 2L
      } else {
        output_width.to_int64() * input_width.to_int64() * 2L
      }
    GatedMlpShape(hidden_size~, intermediate_size~) =>
      if weight_index < 2 {
        intermediate_size.to_int64() * hidden_size.to_int64() * 2L
      } else {
        hidden_size.to_int64() * intermediate_size.to_int64() * 2L
      }
    _ => abort("fixture operation has no weights")
  }
}

///|
fn fixture_operand(
  role : @launch_contract.AotOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count~, alignment~)
}

///|
fn producer_fixture_operands(
  plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
) -> Array[@launch_contract.AotOperand] raise {
  let result : Array[@launch_contract.AotOperand] = []
  for runtime in operation.runtime_inputs() {
    result.push(
      fixture_operand(
        RuntimeInput(runtime),
        fixture_runtime_bytes(runtime, profile),
        4L,
      ),
    )
  }
  let mut weight_index = 0
  for input in operation.inputs() {
    match input {
      ModelInput(reference) =>
        result.push(
          fixture_operand(
            ExternalInput(reference),
            profile.max_query_tokens().to_int64() * 4L,
            4L,
          ),
        )
      Tensor(reference) => {
        result.push(
          fixture_operand(
            WeightInput(reference),
            fixture_weight_bytes(operation, weight_index),
            16L,
          ),
        )
        weight_index += 1
      }
      Activation(reference) =>
        result.push(
          fixture_operand(
            ActivationInput(reference),
            fixture_activation_bytes(plan, reference, profile),
            16L,
          ),
        )
    }
  }
  for output in operation.outputs() {
    result.push(
      fixture_operand(
        ActivationOutput(output),
        fixture_activation_bytes(plan, output, profile),
        16L,
      ),
    )
  }
  match operation.persistent_state() {
    NoPersistentState => ()
    KvCacheReadWrite(layer) => {
      result.push(
        fixture_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          256L,
        ),
      )
      result.push(
        fixture_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          256L,
        ),
      )
    }
  }
  result
}
