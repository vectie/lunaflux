///|
fn execute_operation(
  operation : @plan.PlanOperation,
  token_ids : ArrayView[Int],
  activations : Array[@refkernels.Activation],
  model : PreparedModel,
  limits : ExecutionLimits,
) -> @refkernels.Activation raise ReferenceExecutionError {
  let inputs = operation.inputs()
  let kernel_limits = limits.kernels()
  let result = match (operation.kind(), operation.shape()) {
    (Embedding, EmbeddingShape(vocabulary_size=_, width=_)) => {
      guard inputs[0] is ModelInput(TokenIds) else {
        raise InvalidGraph(UnsupportedInput)
      }
      let table = tensor_input(model, inputs[1])
      @refkernels.embedding(token_ids, table, limits=kernel_limits) catch {
        error => raise Kernel(error)
      }
    }
    (RmsNorm, NormalizationShape(width=_, epsilon~)) => {
      let input = activation_input(activations, inputs[0])
      let weight = tensor_input(model, inputs[1])
      @refkernels.rms_norm(
        input,
        weight,
        epsilon=Float::from_double(epsilon),
        limits=kernel_limits,
      ) catch {
        error => raise Kernel(error)
      }
    }
    (QkvProjection, ProjectionShape(input_width=_, output_width=_)) => {
      let input = activation_input(activations, inputs[0])
      let query = tensor_input(model, inputs[1])
      let key = tensor_input(model, inputs[2])
      let value = tensor_input(model, inputs[3])
      @refkernels.qkv_projection(input, query, key, value, limits=kernel_limits) catch {
        error => raise Kernel(error)
      }
    }
    (
      QkRmsNorm,
      QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, epsilon~),
    ) => {
      let input = activation_input(activations, inputs[0])
      let query_weight = tensor_input(model, inputs[1])
      let key_weight = tensor_input(model, inputs[2])
      qk_rms_norm(
        input,
        query_weight,
        key_weight,
        query_heads~,
        kv_heads~,
        head_dimension~,
        epsilon=Float::from_double(epsilon),
        limits=kernel_limits,
      ) catch {
        error => raise Kernel(error)
      }
    }
    (
      RotaryEmbedding,
      RotaryShape(query_heads~, kv_heads~, head_dimension~, rope_theta~),
    ) => {
      let input = activation_input(activations, inputs[0])
      @refkernels.rotary_embedding(
        input,
        query_heads~,
        kv_heads~,
        head_dimension~,
        position_offset=0,
        rope_theta=Float::from_double(rope_theta),
        limits=kernel_limits,
      ) catch {
        error => raise Kernel(error)
      }
    }
    (CausalAttention, AttentionShape(query_heads~, kv_heads~, head_dimension~)) => {
      let input = activation_input(activations, inputs[0])
      @refkernels.causal_gqa_attention(
        input,
        query_heads~,
        kv_heads~,
        head_dimension~,
        limits=kernel_limits,
      ) catch {
        error => raise Kernel(error)
      }
    }
    (
      OutputProjection
      | LanguageModelHead,
      ProjectionShape(input_width=_, output_width=_),
    ) => {
      let input = activation_input(activations, inputs[0])
      let weight = tensor_input(model, inputs[1])
      @refkernels.dense_projection(input, weight, limits=kernel_limits) catch {
        error => raise Kernel(error)
      }
    }
    (ResidualAdd, ElementwiseShape(width=_)) => {
      let left = activation_input(activations, inputs[0])
      let right = activation_input(activations, inputs[1])
      @refkernels.residual_add(left, right, limits=kernel_limits) catch {
        error => raise Kernel(error)
      }
    }
    (GatedMlp, GatedMlpShape(hidden_size=_, intermediate_size=_)) => {
      let input = activation_input(activations, inputs[0])
      let gate = tensor_input(model, inputs[1])
      let up = tensor_input(model, inputs[2])
      let down = tensor_input(model, inputs[3])
      @refkernels.silu_gated_mlp(input, gate, up, down, limits=kernel_limits) catch {
        error => raise Kernel(error)
      }
    }
    _ => raise InvalidGraph(ShapeMismatch)
  }
  result
}

///|
/// Interprets an immutable semantic model plan using correctness-only kernels.
///
/// This package contains no model-family or device branching. It is an offline
/// Phase 1 correctness oracle and is never a production kernel fallback.
pub fn execute(
  plan : @plan.ModelPlan,
  model : PreparedModel,
  token_ids : ArrayView[Int],
  limits~ : ExecutionLimits,
) -> ExecutionResult raise ReferenceExecutionError {
  if plan.identity() != model.identity() {
    raise InvalidGraph(IdentityMismatch)
  }
  if plan.operations().length() > limits.max_operations() {
    raise InvalidGraph(OperationLimitExceeded)
  }
  if model.tensor_count() > limits.max_tensors() {
    raise InvalidGraph(TensorLimitExceeded)
  }
  if token_ids.is_empty() || token_ids.length() > limits.max_input_tokens() {
    raise InvalidGraph(InputLimitExceeded)
  }
  if token_ids.length() > plan.shape_constraints().max_sequence_tokens() {
    raise InvalidGraph(InputContextExceeded)
  }
  let activations : Array[@refkernels.Activation] = []
  for operation in plan.operations() {
    let activation = execute_operation(
      operation, token_ids, activations, model, limits,
    )
    guard operation.outputs().length() == 1 &&
      operation.outputs()[0].as_int() == activations.length() else {
      raise InvalidGraph(FinalOutputMismatch)
    }
    activations.push(activation)
  }
  guard activations.get(plan.final_output().as_int()) is Some(logits) else {
    raise InvalidGraph(FinalOutputMismatch)
  }
  if logits.rows() != token_ids.length() ||
    logits.columns() != plan.shape_constraints().vocabulary_size() {
    raise InvalidGraph(ShapeMismatch)
  }
  let last_row_start = (logits.rows() - 1) * logits.columns()
  let last_row = logits.values()[last_row_start:]
  let selection = @sampling.greedy_float(last_row, limits.sampling()) catch {
    error => raise Sampling(error)
  }
  {
    identity: plan.identity(),
    logits,
    selection,
    operation_count: plan.operations().length(),
  }
}
