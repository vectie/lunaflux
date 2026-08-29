///|
priv struct ExpectedQwen3Tensor {
  reference : Int
  name : String
  shape : ReadOnlyArray[Int64]
}

///|
fn expected_tensor(
  output : Array[ExpectedQwen3Tensor],
  name : String,
  shape : ArrayView[Int64],
) -> Unit {
  output.push({
    reference: output.length(),
    name,
    shape: ReadOnlyArray::from_array(shape),
  })
}

///|
fn expected_qwen3_tensors(
  spec : @qwen3_spec.Qwen3ModelSpec,
) -> Array[ExpectedQwen3Tensor] {
  let hidden = Int64::from_int(spec.hidden_size())
  let intermediate = Int64::from_int(spec.intermediate_size())
  let vocabulary = Int64::from_int(spec.vocabulary_size())
  let query_width = Int64::from_int(spec.query_projection_width())
  let kv_width = Int64::from_int(spec.kv_projection_width())
  let head_dimension = Int64::from_int(spec.head_dimension())
  let expected : Array[ExpectedQwen3Tensor] = []
  expected_tensor(expected, "model.embed_tokens.weight", [vocabulary, hidden])
  for layer in 0..<spec.layer_count() {
    let prefix = "model.layers.\{layer}"
    expected_tensor(expected, "\{prefix}.input_layernorm.weight", [hidden])
    expected_tensor(expected, "\{prefix}.self_attn.q_proj.weight", [
      query_width, hidden,
    ])
    expected_tensor(expected, "\{prefix}.self_attn.k_proj.weight", [
      kv_width, hidden,
    ])
    expected_tensor(expected, "\{prefix}.self_attn.v_proj.weight", [
      kv_width, hidden,
    ])
    expected_tensor(expected, "\{prefix}.self_attn.q_norm.weight", [
      head_dimension,
    ])
    expected_tensor(expected, "\{prefix}.self_attn.k_norm.weight", [
      head_dimension,
    ])
    expected_tensor(expected, "\{prefix}.self_attn.o_proj.weight", [
      hidden, query_width,
    ])
    expected_tensor(expected, "\{prefix}.post_attention_layernorm.weight", [
      hidden,
    ])
    expected_tensor(expected, "\{prefix}.mlp.gate_proj.weight", [
      intermediate, hidden,
    ])
    expected_tensor(expected, "\{prefix}.mlp.up_proj.weight", [
      intermediate, hidden,
    ])
    expected_tensor(expected, "\{prefix}.mlp.down_proj.weight", [
      hidden, intermediate,
    ])
  }
  expected_tensor(expected, "model.norm.weight", [hidden])
  expected
}

///|
fn dimensions_match(
  tensor : @safetensors.TensorMetadata,
  expected : ExpectedQwen3Tensor,
) -> Unit raise Qwen3WeightsError {
  if tensor.name() != expected.name {
    raise InvalidTensor(reference=expected.reference, TensorName)
  }
  if tensor.dtype() != BFloat16 {
    raise InvalidTensor(reference=expected.reference, TensorDType)
  }
  let actual = tensor.shape()
  if actual.length() != expected.shape.length() {
    raise InvalidTensor(reference=expected.reference, TensorRank)
  }
  let mut expected_bytes = 2L
  for axis, dimension in expected.shape {
    if actual[axis] != dimension {
      raise InvalidTensor(reference=expected.reference, TensorDimension)
    }
    expected_bytes = expected_bytes * dimension
  }
  if tensor.data_start() < 0L ||
    tensor.data_end() < tensor.data_start() ||
    tensor.byte_length() != expected_bytes {
    raise InvalidTensor(reference=expected.reference, TensorByteRange)
  }
}

///|
fn known_tensor_name(
  name : StringView,
  expected : ArrayView[ExpectedQwen3Tensor],
) -> Bool {
  if name == "lm_head.weight" {
    return true
  }
  for tensor in expected {
    if tensor.name[:] == name {
      return true
    }
  }
  false
}

///|
fn validate_source_vocabulary(
  tensors : @safetensors.SafetensorsMetadata,
  expected : ArrayView[ExpectedQwen3Tensor],
) -> Unit raise Qwen3WeightsError {
  for index in 0..<tensors.tensor_count() {
    guard tensors.tensor(index) is Some(tensor) else {
      raise Mismatch(TensorCount)
    }
    if !known_tensor_name(tensor.name(), expected) {
      raise UnknownTensor(index~)
    }
  }
}

///|
fn tied_output_tensor(
  tensors : @safetensors.SafetensorsMetadata,
  embedding : @safetensors.TensorMetadata,
) -> @safetensors.TensorMetadata? raise Qwen3WeightsError {
  match tensors.find_tensor("lm_head.weight") {
    None => None
    Some(head) => {
      let expected : ExpectedQwen3Tensor = {
        reference: 0,
        name: "lm_head.weight",
        shape: embedding.shape(),
      }
      dimensions_match(head, expected) catch {
        _ => raise InvalidTensor(reference=0, TiedOutput)
      }
      if head.byte_length() != embedding.byte_length() {
        raise InvalidTensor(reference=0, TiedOutput)
      }
      Some(head)
    }
  }
}

///|
/// Binds the exact dense Qwen3 BF16 safetensors vocabulary in future semantic
/// `TensorRef` order without minting a model-plan identity.
///
/// Per layer the order is input norm; Q, K, V projections; Q and K per-head
/// norms; output projection; post-attention norm; gate, up, and down
/// projections. The final norm follows all layers. The language-model head is
/// the embedding table's exact reference-zero storage alias. A source artifact
/// may omit `lm_head.weight`, or carry one validated redundant tied copy; that
/// copy is never assigned a second reference or counted for materialization.
fn bind_qwen3_metadata(
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  tensors : @safetensors.SafetensorsMetadata,
) -> Qwen3WeightBindings raise Qwen3WeightsError {
  let spec = metadata.spec()
  if !spec.tied_embeddings() {
    raise Mismatch(TiedOutput)
  }
  let expected = expected_qwen3_tensors(spec)
  validate_source_vocabulary(tensors, expected)
  let physical_count = tensors.tensor_count()
  let bindings : Array[Qwen3TensorBinding] = []
  let mut materialized_bytes = 0L
  for descriptor in expected {
    guard tensors.find_tensor(descriptor.name) is Some(tensor) else {
      raise MissingTensor(reference=descriptor.reference)
    }
    dimensions_match(tensor, descriptor)
    let reference = @plan.TensorRef::new(descriptor.reference) catch {
      _ => raise InvalidTensor(reference=descriptor.reference, TensorCount)
    }
    bindings.push({ reference, tensor })
    materialized_bytes = materialized_bytes + tensor.byte_length()
  }
  let embedding = bindings[0].tensor
  let redundant_lm_head = tied_output_tensor(tensors, embedding)
  if physical_count != expected.length() &&
    physical_count != expected.length() + 1 {
    raise Mismatch(TensorCount)
  }
  if physical_count == expected.length() + 1 && redundant_lm_head is None {
    raise Mismatch(TensorCount)
  }
  {
    content_digest: metadata.content_digest(),
    bindings: ReadOnlyArray::from_array(bindings),
    source_tensor_count: physical_count,
    source_bytes: tensors.materialized_tensor_bytes(),
    materialized_bytes,
    redundant_lm_head,
  }
}

///|
/// Binds metadata-only Qwen3 sources which omit redundant tied-output bytes.
///
/// Metadata can prove shape and range but cannot prove that a physical
/// `lm_head.weight` is byte-identical to the embedding. Such sources must use
/// `inspect_source_safetensors_file`, which performs the actual byte proof
/// before returning bindings with a reference-zero LM-head alias.
pub fn bind(
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  tensors : @safetensors.SafetensorsMetadata,
) -> Qwen3WeightBindings raise Qwen3WeightsError {
  let bindings = bind_qwen3_metadata(metadata, tensors)
  if bindings.redundant_lm_head_tensor() is Some(_) {
    raise InvalidTensor(reference=0, TiedOutputByteEquality)
  }
  bindings
}
