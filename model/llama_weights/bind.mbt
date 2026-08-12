///|
fn checked_tensor_count(layer_count : Int) -> Int {
  // `LlamaModelSpec` dimensions are positive Int values. The model reader and
  // spec constructor bound practical layer counts before this startup-only
  // multiplication; use Int64 here so malformed metadata cannot wrap Int.
  let count = 3L + Int64::from_int(layer_count) * 9L
  if count > 2147483647L {
    2147483647
  } else {
    count.to_int()
  }
}

///|
fn dimensions_match(
  actual : ReadOnlyArray[Int64],
  expected : ArrayView[Int64],
  reference : Int,
) -> Unit raise LlamaWeightsError {
  if actual.length() != expected.length() {
    raise ShapeRankMismatch(reference~)
  }
  for axis, dimension in expected {
    if actual[axis] != dimension {
      raise ShapeDimensionMismatch(reference~, axis~)
    }
  }
}

///|
fn append_binding(
  output : Array[TensorBinding],
  tensors : @safetensors.SafetensorsMetadata,
  name : StringView,
  shape : ArrayView[Int64],
) -> Unit raise LlamaWeightsError {
  let reference = output.length()
  guard tensors.find_tensor(name) is Some(tensor) else {
    raise MissingTensor(reference~)
  }
  dimensions_match(tensor.shape(), shape, reference)
  let tensor_reference = try! @plan.TensorRef::new(reference)
  output.push({ reference: tensor_reference, tensor })
}

///|
/// Binds the exact dense Llama v1 tensor vocabulary to the same `TensorRef`
/// order emitted by `model/llama.build`.
///
/// The input metadata is already byte-range and dtype validated. This pass
/// adds architecture-specific name, rank, dimension, completeness, and order
/// validation before any device allocation is permitted.
pub fn bind(
  model : @spec.LlamaModelMetadata,
  tensors : @safetensors.SafetensorsMetadata,
) -> LlamaWeightBindings raise LlamaWeightsError {
  let spec = model.spec()
  let expected_count = checked_tensor_count(spec.layer_count())
  if tensors.tensor_count() != expected_count {
    raise TensorCountMismatch(
      expected=expected_count,
      actual=tensors.tensor_count(),
    )
  }
  let hidden = Int64::from_int(spec.hidden_size())
  let intermediate = Int64::from_int(spec.intermediate_size())
  let vocabulary = Int64::from_int(spec.vocabulary_size())
  let kv_width = Int64::from_int(spec.kv_head_count() * spec.head_dimension())
  let bindings : Array[TensorBinding] = []
  append_binding(bindings, tensors, "model.embed_tokens.weight", [
    vocabulary, hidden,
  ])
  for layer in 0..<spec.layer_count() {
    let prefix = "model.layers.\{layer}"
    append_binding(bindings, tensors, "\{prefix}.input_layernorm.weight", [
      hidden,
    ])
    append_binding(bindings, tensors, "\{prefix}.self_attn.q_proj.weight", [
      hidden, hidden,
    ])
    append_binding(bindings, tensors, "\{prefix}.self_attn.k_proj.weight", [
      kv_width, hidden,
    ])
    append_binding(bindings, tensors, "\{prefix}.self_attn.v_proj.weight", [
      kv_width, hidden,
    ])
    append_binding(bindings, tensors, "\{prefix}.self_attn.o_proj.weight", [
      hidden, hidden,
    ])
    append_binding(
      bindings,
      tensors,
      "\{prefix}.post_attention_layernorm.weight",
      [hidden],
    )
    append_binding(bindings, tensors, "\{prefix}.mlp.gate_proj.weight", [
      intermediate, hidden,
    ])
    append_binding(bindings, tensors, "\{prefix}.mlp.up_proj.weight", [
      intermediate, hidden,
    ])
    append_binding(bindings, tensors, "\{prefix}.mlp.down_proj.weight", [
      hidden, intermediate,
    ])
  }
  append_binding(bindings, tensors, "model.norm.weight", [hidden])
  append_binding(bindings, tensors, "lm_head.weight", [vocabulary, hidden])
  {
    identity: model.identity(),
    bindings: ReadOnlyArray::from_array(bindings),
    materialized_bytes: tensors.materialized_tensor_bytes(),
  }
}
