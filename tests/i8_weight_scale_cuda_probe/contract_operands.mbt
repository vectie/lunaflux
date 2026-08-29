///|
fn probe_operand(
  role : @launch_contract.AotOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count~, alignment~)
}

///|
fn probe_tensor_bytes(
  plan : @plan.ModelPlan,
  reference : @plan.TensorRef,
) -> Int64 raise {
  guard plan.tensor(reference) is Some(tensor) else {
    raise I8PhysicalProbeFailure("probe tensor is absent")
  }
  @numeric_contract.storage_byte_length(
    tensor.storage().dtype(),
    tensor.shape(),
  )
}

///|
fn probe_runtime_bytes(
  runtime : @plan.OperationRuntimeInput,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  match runtime {
    StepCounts => 20L
    QueryPositions => profile.max_query_tokens().to_int64() * 4L
    QueryRowOffsets | PageTableOffsets =>
      (profile.max_query_rows() + 1).to_int64() * 4L
    SequenceLengths => profile.max_query_rows().to_int64() * 4L
    PageTablePageIndices => profile.max_page_table_entries().to_int64() * 4L
  }
}

///|
fn probe_activation_input_width(
  operation : @plan.PlanOperation,
) -> Int64 raise I8PhysicalProbeFailure {
  match (operation.kind(), operation.shape()) {
    (RmsNorm, NormalizationShape(width~, ..)) => width.to_int64()
    (QkvProjection, ProjectionShape(input_width~, ..)) => input_width.to_int64()
    (RotaryEmbedding, RotaryShape(query_heads~, kv_heads~, head_dimension~, ..)) =>
      (query_heads + 2 * kv_heads).to_int64() * head_dimension.to_int64()
    (CausalAttention, AttentionShape(query_heads~, kv_heads~, head_dimension~)) =>
      (query_heads + 2 * kv_heads).to_int64() * head_dimension.to_int64()
    (OutputProjection | LanguageModelHead, ProjectionShape(input_width~, ..)) =>
      input_width.to_int64()
    (ResidualAdd, ElementwiseShape(width~)) => width.to_int64()
    (GatedMlp, GatedMlpShape(hidden_size~, ..)) => hidden_size.to_int64()
    _ => raise I8PhysicalProbeFailure("unsupported probe activation input")
  }
}

///|
fn probe_activation_output_width(
  operation : @plan.PlanOperation,
) -> Int64 raise I8PhysicalProbeFailure {
  match (operation.kind(), operation.shape()) {
    (Embedding, EmbeddingShape(width~, ..)) => width.to_int64()
    (RmsNorm, NormalizationShape(width~, ..)) => width.to_int64()
    (QkvProjection, ProjectionShape(output_width~, ..)) =>
      output_width.to_int64()
    (RotaryEmbedding, RotaryShape(query_heads~, kv_heads~, head_dimension~, ..)) =>
      (query_heads + 2 * kv_heads).to_int64() * head_dimension.to_int64()
    (CausalAttention, AttentionShape(query_heads~, head_dimension~, ..)) =>
      query_heads.to_int64() * head_dimension.to_int64()
    (OutputProjection | LanguageModelHead, ProjectionShape(output_width~, ..)) =>
      output_width.to_int64()
    (ResidualAdd, ElementwiseShape(width~)) => width.to_int64()
    (GatedMlp, GatedMlpShape(hidden_size~, ..)) => hidden_size.to_int64()
    _ => raise I8PhysicalProbeFailure("unsupported probe activation output")
  }
}

///|
fn probe_operands(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
) -> Array[@launch_contract.AotOperand] raise {
  let result : Array[@launch_contract.AotOperand] = []
  for runtime in operation.runtime_inputs() {
    result.push(
      probe_operand(
        RuntimeInput(runtime),
        probe_runtime_bytes(runtime, profile),
        4L,
      ),
    )
  }
  for input in operation.inputs() {
    match input {
      ModelInput(reference) =>
        result.push(
          probe_operand(
            ExternalInput(reference),
            profile.max_query_tokens().to_int64() * 4L,
            4L,
          ),
        )
      Activation(reference) =>
        result.push(
          probe_operand(
            ActivationInput(reference),
            profile.max_query_tokens().to_int64() *
            probe_activation_input_width(operation) *
            2L,
            16L,
          ),
        )
      Tensor(reference) => {
        result.push(
          probe_operand(
            WeightInput(reference),
            probe_tensor_bytes(plan, reference),
            16L,
          ),
        )
        guard plan.tensor(reference) is Some(tensor) else {
          raise I8PhysicalProbeFailure("probe weight is absent")
        }
        match tensor.scale() {
          Some(scale) =>
            result.push(
              probe_operand(
                WeightScaleInput(reference, scale),
                probe_tensor_bytes(plan, scale.tensor()),
                16L,
              ),
            )
          None => ()
        }
      }
    }
  }
  for output in operation.outputs() {
    result.push(
      probe_operand(
        ActivationOutput(output),
        profile.max_query_tokens().to_int64() *
        probe_activation_output_width(operation) *
        2L,
        16L,
      ),
    )
  }
  match operation.persistent_state() {
    NoPersistentState => ()
    KvCacheReadWrite(layer) => {
      result.push(
        probe_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
      result.push(
        probe_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
    }
  }
  if operation.workspace_bytes() > 0L {
    result.push(probe_operand(Workspace, operation.workspace_bytes(), 16L))
  }
  result
}

///|
fn probe_launch_dimensions() -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn probe_launch_contracts(
  plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
  layout : @device_layout.DeviceKvLayout,
  module_digest : @catalog.AotArtifactDigest,
) -> @launch_contract.PagedV4LaunchContractSet raise {
  let profile = probe_profile()
  let inputs : Array[@launch_contract.PagedV4AotLaunchContractInput] = []
  for index, operation in plan.operations() {
    inputs.push(
      @launch_contract.PagedV4AotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=operation.id(),
        entry_point=probe_entry_point(
          module_digest,
          probe_representative_operation(plan, index),
        ),
        dimensions=probe_launch_dimensions(),
        operands=probe_operands(plan, layout, operation, profile),
      ),
    )
  }
  @launch_contract.admit_paged_graph_v4(
    plan,
    resolved,
    layout,
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=plan.operations().length(),
      max_operands_per_contract=16,
    ),
  )
}
