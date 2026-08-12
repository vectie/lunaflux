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
fn append_operation(
  operations : Array[@plan.PlanOperation],
  kind : @plan.OperationKind,
  inputs : ArrayView[@plan.ValueRef],
  shape : @plan.OperationShape,
  capability : @plan.KernelCapabilityId,
) -> @plan.ActivationRef raise @plan.PlanValidationError {
  let output = @plan.ActivationRef::new(operations.length())
  let operation = @plan.PlanOperation::new(
    id=@plan.OperationId::new(operations.length()),
    kind~,
    inputs~,
    outputs=[output],
    shape~,
    required_capability=capability,
    workspace_bytes=0L,
  )
  operations.push(operation)
  output
}

///|
/// Builds the immutable dense Llama v1 semantic plan.
///
/// All model-family branching terminates in this package. The returned graph
/// contains no model-family, device, vendor, allocation, or kernel-artifact
/// switch.
pub fn build(
  metadata : @spec.LlamaModelMetadata,
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  let spec = metadata.spec()
  let hidden = spec.hidden_size()
  let intermediate = spec.intermediate_size()
  let query_heads = spec.attention_head_count()
  let kv_heads = spec.kv_head_count()
  let head_dimension = spec.head_dimension()
  let constraints = @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=spec.context_length(),
    vocabulary_size=spec.vocabulary_size(),
    hidden_size=hidden,
    intermediate_size=intermediate,
    attention_head_count=query_heads,
    kv_head_count=kv_heads,
    head_dimension~,
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
  let mut next_tensor = 0
  let mut hidden_state = append_operation(
    operations,
    Embedding,
    [ModelInput(TokenIds), tensor(next_tensor)],
    EmbeddingShape(vocabulary_size=spec.vocabulary_size(), width=hidden),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  next_tensor += 1
  for _layer in 0..<spec.layer_count() {
    let residual_input = hidden_state
    let attention_norm = append_operation(
      operations,
      RmsNorm,
      [activation(hidden_state), tensor(next_tensor)],
      NormalizationShape(width=hidden, epsilon=spec.rms_norm_epsilon()),
      @plan.KernelCapabilityId::rms_norm(),
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
    )
    next_tensor += 3
    let rotated = append_operation(
      operations,
      RotaryEmbedding,
      [activation(projected)],
      RotaryShape(
        query_heads~,
        kv_heads~,
        head_dimension~,
        rope_theta=spec.rope_theta(),
      ),
      @plan.KernelCapabilityId::rotary_embedding(),
    )
    let attended = append_operation(
      operations,
      CausalAttention,
      [activation(rotated)],
      AttentionShape(query_heads~, kv_heads~, head_dimension~),
      @plan.KernelCapabilityId::causal_attention(),
    )
    let projected_output = append_operation(
      operations,
      OutputProjection,
      [activation(attended), tensor(next_tensor)],
      ProjectionShape(input_width=hidden, output_width=hidden),
      @plan.KernelCapabilityId::dense_projection(),
    )
    next_tensor += 1
    let attention_residual = append_operation(
      operations,
      ResidualAdd,
      [activation(residual_input), activation(projected_output)],
      ElementwiseShape(width=hidden),
      @plan.KernelCapabilityId::residual_add(),
    )
    let mlp_norm = append_operation(
      operations,
      RmsNorm,
      [activation(attention_residual), tensor(next_tensor)],
      NormalizationShape(width=hidden, epsilon=spec.rms_norm_epsilon()),
      @plan.KernelCapabilityId::rms_norm(),
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
    )
    next_tensor += 3
    hidden_state = append_operation(
      operations,
      ResidualAdd,
      [activation(attention_residual), activation(mlp_output)],
      ElementwiseShape(width=hidden),
      @plan.KernelCapabilityId::residual_add(),
    )
  }
  let final_norm = append_operation(
    operations,
    RmsNorm,
    [activation(hidden_state), tensor(next_tensor)],
    NormalizationShape(width=hidden, epsilon=spec.rms_norm_epsilon()),
    @plan.KernelCapabilityId::rms_norm(),
  )
  next_tensor += 1
  let logits = append_operation(
    operations,
    LanguageModelHead,
    [activation(final_norm), tensor(next_tensor)],
    ProjectionShape(input_width=hidden, output_width=spec.vocabulary_size()),
    @plan.KernelCapabilityId::dense_projection(),
  )
  @plan.ModelPlan::new(
    identity=metadata.identity(),
    shape_constraints=constraints,
    workspace_bounds=workspace,
    operations~,
    final_output=logits,
  )
}
