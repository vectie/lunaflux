///|
fn checked_product(
  left : Int64,
  right : Int64,
) -> Int64 raise @plan.PlanValidationError {
  if left < 0L ||
    right < 0L ||
    (right != 0L && left > 9223372036854775807L / right) {
    raise InvalidWorkspaceBound
  }
  left * right
}

///|
fn tensor(reference : Int) -> @plan.ValueRef raise @plan.PlanValidationError {
  Tensor(@plan.TensorRef::new(reference))
}

///|
fn activation(reference : @plan.ActivationRef) -> @plan.ValueRef {
  Activation(reference)
}

///|
priv enum LlamaOperationWorkspacePolicy {
  LegacyOperationWorkspaceV1
  Fp8DynamicActivationWorkspaceV2
}

///|
fn llama_operation_workspace_bytes(
  kind : @plan.OperationKind,
  policy : LlamaOperationWorkspacePolicy,
) -> Int64 {
  match (policy, kind) {
    (Fp8DynamicActivationWorkspaceV2, GatedMlp) => 8L
    (
      Fp8DynamicActivationWorkspaceV2,
      QkvProjection
      | OutputProjection
      | LanguageModelHead,
    ) => 4L
    _ => 0L
  }
}

///|
fn append_operation(
  operations : Array[@plan.PlanOperation],
  kind : @plan.OperationKind,
  inputs : ArrayView[@plan.ValueRef],
  shape : @plan.OperationShape,
  capability : @plan.KernelCapabilityId,
  runtime_inputs : ArrayView[@plan.OperationRuntimeInput],
  workspace_policy : LlamaOperationWorkspacePolicy,
) -> @plan.ActivationRef raise @plan.PlanValidationError {
  let output = @plan.ActivationRef::new(operations.length())
  let workspace_bytes = llama_operation_workspace_bytes(kind, workspace_policy)
  let operation = if runtime_inputs.is_empty() {
    @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(operations.length()),
      kind~,
      inputs~,
      outputs=[output],
      shape~,
      required_capability=capability,
      workspace_bytes~,
    )
  } else {
    @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(operations.length()),
      kind~,
      inputs~,
      outputs=[output],
      shape~,
      required_capability=capability,
      workspace_bytes~,
      runtime_inputs~,
    )
  }
  operations.push(operation)
  output
}

///|
fn append_layer_operation(
  operations : Array[@plan.PlanOperation],
  layer : @plan.KvLayerId,
  kind : @plan.OperationKind,
  inputs : ArrayView[@plan.ValueRef],
  shape : @plan.OperationShape,
  capability : @plan.KernelCapabilityId,
  runtime_inputs : ArrayView[@plan.OperationRuntimeInput],
  persistent_state : @plan.PersistentStateAccess,
) -> @plan.ActivationRef raise @plan.PlanValidationError {
  let output = @plan.ActivationRef::new(operations.length())
  operations.push(
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(operations.length()),
      kind~,
      inputs~,
      outputs=[output],
      shape~,
      required_capability=capability,
      workspace_bytes=0L,
      kv_layer=layer,
      runtime_inputs~,
      persistent_state~,
    ),
  )
  output
}

///|
/// Builds one immutable dense Llama semantic plan.
///
/// All model-family branching terminates in this package. The returned graph
/// contains no model-family, device, vendor, allocation, or kernel-artifact
/// switch. Paged plans bind their exact admitted batch-row ceiling into both
/// shape constraints and the self-authenticated plan identity.
fn build_with_execution(
  metadata : @spec.LlamaModelMetadata,
  kv_execution : @plan.ModelKvExecution,
  max_batch_rows : Int,
  numeric_schema : @numeric_contract.ModelNumericSchema,
  workspace_policy : LlamaOperationWorkspacePolicy,
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  let spec = metadata.spec()
  let hidden = spec.hidden_size()
  let intermediate = spec.intermediate_size()
  let query_heads = spec.attention_head_count()
  let kv_heads = spec.kv_head_count()
  let head_dimension = spec.head_dimension()
  let constraints = @plan.ShapeConstraints::new(
    max_batch_rows~,
    max_sequence_tokens=spec.context_length(),
    vocabulary_size=spec.vocabulary_size(),
    hidden_size=hidden,
    intermediate_size=intermediate,
    attention_head_count=query_heads,
    kv_head_count=kv_heads,
    head_dimension~,
  )
  let kv_cache_geometry = @plan.KvCacheGeometry::new(
    layer_count=spec.layer_count(),
    kv_head_count=kv_heads,
    head_dimension~,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
  let kv_width = kv_heads * head_dimension
  let qkv_width = hidden + 2 * kv_width
  let qkv_scratch_width = Int64::from_int(qkv_width)
  let mlp_scratch_width = checked_product(Int64::from_int(intermediate), 2L)
  let scratch_width = qkv_scratch_width.max(mlp_scratch_width)
  let scratch_rows = checked_product(
    Int64::from_int(constraints.max_batch_rows()),
    Int64::from_int(constraints.max_sequence_tokens()),
  )
  let scratch_elements = checked_product(scratch_width, scratch_rows)
  let workspace = @plan.WorkspaceBounds::new(
    max_temporary_bytes=checked_product(scratch_elements, 2L),
  )
  let operations : Array[@plan.PlanOperation] = []
  let live_runtime : Array[@plan.OperationRuntimeInput] = match kv_execution {
    StatelessFullContext => []
    PagedKeyValue => [StepCounts]
  }
  let mut next_tensor = 0
  let mut hidden_state = append_operation(
    operations,
    Embedding,
    [ModelInput(TokenIds), tensor(next_tensor)],
    EmbeddingShape(vocabulary_size=spec.vocabulary_size(), width=hidden),
    @plan.KernelCapabilityId::embedding_lookup(),
    live_runtime,
    workspace_policy,
  )
  next_tensor += 1
  for layer_index in 0..<spec.layer_count() {
    let layer = @plan.KvLayerId::new(layer_index)
    let residual_input = hidden_state
    let attention_norm = append_operation(
      operations,
      RmsNorm,
      [activation(hidden_state), tensor(next_tensor)],
      NormalizationShape(width=hidden, epsilon=spec.rms_norm_epsilon()),
      @plan.KernelCapabilityId::rms_norm(),
      live_runtime,
      workspace_policy,
    )
    next_tensor += 1
    let projected = append_operation(
      operations,
      QkvProjection,
      [
        activation(attention_norm),
        tensor(next_tensor),
        tensor(next_tensor + 1),
        tensor(next_tensor + 2),
      ],
      ProjectionShape(input_width=hidden, output_width=qkv_width),
      @plan.KernelCapabilityId::qkv_projection(),
      live_runtime,
      workspace_policy,
    )
    next_tensor += 3
    let rotary_runtime = match kv_execution {
      StatelessFullContext => []
      PagedKeyValue => [@plan.StepCounts, QueryPositions]
    }
    let rotated = append_layer_operation(
      operations,
      layer,
      RotaryEmbedding,
      [activation(projected)],
      RotaryShape(
        query_heads~,
        kv_heads~,
        head_dimension~,
        rope_theta=spec.rope_theta(),
      ),
      match kv_execution {
        StatelessFullContext => @plan.KernelCapabilityId::rotary_embedding()
        PagedKeyValue => @plan.KernelCapabilityId::positioned_rotary_embedding()
      },
      rotary_runtime,
      NoPersistentState,
    )
    let attention_runtime = match kv_execution {
      StatelessFullContext => []
      PagedKeyValue =>
        [
          @plan.StepCounts,
          QueryPositions,
          QueryRowOffsets,
          SequenceLengths,
          PageTableOffsets,
          PageTablePageIndices,
        ]
    }
    let persistent_state = match kv_execution {
      StatelessFullContext => @plan.NoPersistentState
      PagedKeyValue => KvCacheReadWrite(layer)
    }
    let attended = append_layer_operation(
      operations,
      layer,
      CausalAttention,
      [activation(rotated)],
      AttentionShape(query_heads~, kv_heads~, head_dimension~),
      match kv_execution {
        StatelessFullContext => @plan.KernelCapabilityId::causal_attention()
        PagedKeyValue => @plan.KernelCapabilityId::paged_causal_attention()
      },
      attention_runtime,
      persistent_state,
    )
    let projected_output = append_operation(
      operations,
      OutputProjection,
      [activation(attended), tensor(next_tensor)],
      ProjectionShape(input_width=hidden, output_width=hidden),
      @plan.KernelCapabilityId::dense_projection(),
      live_runtime,
      workspace_policy,
    )
    next_tensor += 1
    let attention_residual = append_operation(
      operations,
      ResidualAdd,
      [activation(residual_input), activation(projected_output)],
      ElementwiseShape(width=hidden),
      @plan.KernelCapabilityId::residual_add(),
      live_runtime,
      workspace_policy,
    )
    let mlp_norm = append_operation(
      operations,
      RmsNorm,
      [activation(attention_residual), tensor(next_tensor)],
      NormalizationShape(width=hidden, epsilon=spec.rms_norm_epsilon()),
      @plan.KernelCapabilityId::rms_norm(),
      live_runtime,
      workspace_policy,
    )
    next_tensor += 1
    let mlp_output = append_operation(
      operations,
      GatedMlp,
      [
        activation(mlp_norm),
        tensor(next_tensor),
        tensor(next_tensor + 1),
        tensor(next_tensor + 2),
      ],
      GatedMlpShape(hidden_size=hidden, intermediate_size=intermediate),
      @plan.KernelCapabilityId::gated_mlp(),
      live_runtime,
      workspace_policy,
    )
    next_tensor += 3
    hidden_state = append_operation(
      operations,
      ResidualAdd,
      [activation(attention_residual), activation(mlp_output)],
      ElementwiseShape(width=hidden),
      @plan.KernelCapabilityId::residual_add(),
      live_runtime,
      workspace_policy,
    )
  }
  let final_norm = append_operation(
    operations,
    RmsNorm,
    [activation(hidden_state), tensor(next_tensor)],
    NormalizationShape(width=hidden, epsilon=spec.rms_norm_epsilon()),
    @plan.KernelCapabilityId::rms_norm(),
    live_runtime,
    workspace_policy,
  )
  next_tensor += 1
  let logits = append_operation(
    operations,
    LanguageModelHead,
    [activation(final_norm), tensor(next_tensor)],
    ProjectionShape(input_width=hidden, output_width=spec.vocabulary_size()),
    @plan.KernelCapabilityId::dense_projection(),
    match kv_execution {
      StatelessFullContext => []
      PagedKeyValue => [StepCounts, QueryRowOffsets]
    },
    workspace_policy,
  )
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=metadata.content_digest(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=constraints,
    kv_cache_geometry~,
    workspace_bounds=workspace,
    operations~,
    final_output=logits,
    kv_execution~,
    numeric_schema~,
  )
}

///|
/// Builds the immutable full-context dense Llama plan used by the current
/// correctness and Phase-1 execution paths. Decoder-layer identities are
/// explicit, but no operation claims persistent KV state.
pub fn build(
  metadata : @spec.LlamaModelMetadata,
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  let spec = metadata.spec()
  let numeric_schema = spec.numeric_schema()
  build_with_execution(
    metadata,
    StatelessFullContext,
    1,
    numeric_schema,
    LegacyOperationWorkspaceV1,
  )
}

///|
/// Builds the full-live semantic paged-KV graph for cached execution planning.
///
/// Every operation consumes `StepCounts`; rotary and attention additionally
/// consume their exact positioned-query and page-table inputs. This constructs
/// no device profile and proves no kernel support. Catalog and executor
/// admission must reject the plan until they bind the complete graph. The
/// optional envelope defaults to the canonical single-row compatibility plan;
/// every larger envelope derives a distinct authenticated plan identity.
pub fn build_paged(
  metadata : @spec.LlamaModelMetadata,
  batch? : LlamaPagedBatchEnvelope = LlamaPagedBatchEnvelope::single_row(),
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  let spec = metadata.spec()
  let numeric_schema = spec.numeric_schema()
  build_with_execution(
    metadata,
    PagedKeyValue,
    batch.max_batch_rows(),
    numeric_schema,
    LegacyOperationWorkspaceV1,
  )
}
