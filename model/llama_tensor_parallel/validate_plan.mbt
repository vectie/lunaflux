///|
fn llama_tp_require_operation(
  operations : ReadOnlyArray[@plan.PlanOperation],
  index : Int,
  kind : @plan.OperationKind,
) -> @plan.PlanOperation raise LlamaTensorParallelError {
  if index < 0 ||
    index >= operations.length() ||
    operations[index].kind() != kind {
    raise UnsupportedPlan(operation=index)
  }
  operations[index]
}

///|
fn llama_tp_require_tensor_input(
  operation : @plan.PlanOperation,
  input_index : Int,
  reference : Int,
) -> Unit raise LlamaTensorParallelError {
  let inputs = operation.inputs()
  if input_index < 0 || input_index >= inputs.length() {
    raise UnsupportedPlan(operation=operation.id().as_int())
  }
  match inputs[input_index] {
    Tensor(actual) if actual.as_int() == reference => ()
    _ => raise UnsupportedPlan(operation=operation.id().as_int())
  }
}

///|
fn llama_tp_require_no_tensor_inputs(
  operation : @plan.PlanOperation,
) -> Unit raise LlamaTensorParallelError {
  for input in operation.inputs() {
    if input is Tensor(_) {
      raise UnsupportedPlan(operation=operation.id().as_int())
    }
  }
}

///|
fn llama_tp_validate_dense_plan(
  model : @plan.ModelPlan,
  bindings : @llama_weights.LlamaWeightBindings,
) -> Int raise LlamaTensorParallelError {
  if model.identity().content() != bindings.identity().content() {
    raise IdentityMismatch
  }
  let layer_count = model.kv_cache_geometry().layer_count()
  let expected_operations = 3 + layer_count * 9
  let expected_tensors = 3 + layer_count * 9
  let operations = model.operations()
  if operations.length() != expected_operations ||
    bindings.length() != expected_tensors {
    raise UnsupportedPlan(operation=operations.length())
  }
  let embedding = llama_tp_require_operation(operations, 0, Embedding)
  llama_tp_require_tensor_input(embedding, 1, 0)
  for layer in 0..<layer_count {
    let operation_base = 1 + layer * 9
    let tensor_base = 1 + layer * 9
    let attention_norm = llama_tp_require_operation(
      operations,
      operation_base,
      RmsNorm,
    )
    llama_tp_require_tensor_input(attention_norm, 1, tensor_base)
    let qkv = llama_tp_require_operation(
      operations,
      operation_base + 1,
      QkvProjection,
    )
    for offset in 0..<3 {
      llama_tp_require_tensor_input(qkv, 1 + offset, tensor_base + 1 + offset)
    }
    llama_tp_require_no_tensor_inputs(
      llama_tp_require_operation(
        operations,
        operation_base + 2,
        RotaryEmbedding,
      ),
    )
    llama_tp_require_no_tensor_inputs(
      llama_tp_require_operation(
        operations,
        operation_base + 3,
        CausalAttention,
      ),
    )
    let output = llama_tp_require_operation(
      operations,
      operation_base + 4,
      OutputProjection,
    )
    llama_tp_require_tensor_input(output, 1, tensor_base + 4)
    llama_tp_require_no_tensor_inputs(
      llama_tp_require_operation(operations, operation_base + 5, ResidualAdd),
    )
    let mlp_norm = llama_tp_require_operation(
      operations,
      operation_base + 6,
      RmsNorm,
    )
    llama_tp_require_tensor_input(mlp_norm, 1, tensor_base + 5)
    let mlp = llama_tp_require_operation(
      operations,
      operation_base + 7,
      GatedMlp,
    )
    for offset in 0..<3 {
      llama_tp_require_tensor_input(mlp, 1 + offset, tensor_base + 6 + offset)
    }
    llama_tp_require_no_tensor_inputs(
      llama_tp_require_operation(operations, operation_base + 8, ResidualAdd),
    )
  }
  let final_operation = 1 + layer_count * 9
  let final_tensor = 1 + layer_count * 9
  let final_norm = llama_tp_require_operation(
    operations,
    final_operation,
    RmsNorm,
  )
  llama_tp_require_tensor_input(final_norm, 1, final_tensor)
  let head = llama_tp_require_operation(
    operations,
    final_operation + 1,
    LanguageModelHead,
  )
  llama_tp_require_tensor_input(head, 1, final_tensor + 1)
  layer_count
}

///|
fn llama_tp_require_binding_shape(
  bindings : @llama_weights.LlamaWeightBindings,
  reference : Int,
  rows : Int64,
  columns : Int64,
) -> @safetensors.TensorMetadata raise LlamaTensorParallelError {
  let tensor_reference = @plan.TensorRef::new(reference) catch {
    _ => raise UnsupportedBinding(reference~)
  }
  guard bindings.binding(tensor_reference) is Some(binding) else {
    raise UnsupportedBinding(reference~)
  }
  let tensor = binding.tensor()
  let shape = tensor.shape()
  let valid = if rows == 1L {
    shape.length() == 1 && shape[0] == columns
  } else {
    shape.length() == 2 && shape[0] == rows && shape[1] == columns
  }
  if !valid || tensor.dtype() != BFloat16 {
    raise UnsupportedBinding(reference~)
  }
  tensor
}
