///|
/// Validates the complete safetensors envelope and every bound tensor range,
/// then visits zero-copy tensor views in semantic `TensorRef` order.
///
/// Validation is a complete first pass: the visitor is never invoked for a
/// malformed envelope or range set. Validation failures are returned as a
/// typed `Err`; visitor errors propagate unchanged, so a final-destination
/// implementation retains its own typed failure vocabulary.
pub fn visit(
  bindings : @llama_weights.LlamaWeightBindings,
  file : BytesView,
  limits~ : MaterializationLimits,
  visitor : (TensorChunk) -> Unit raise?,
) -> Result[MaterializationReport, MaterializationError] raise? {
  let prepared = Ok(prepare(bindings, file, limits)) catch {
    error => Err(error)
  }
  match prepared {
    Err(error) => Err(error)
    Ok(tensors) => {
      let receipts : Array[TensorReceipt] = []
      for prepared_tensor in tensors {
        let binding = prepared_tensor.binding
        let bytes = prepared_tensor.bytes
        let chunk : TensorChunk = {
          reference: binding.reference(),
          tensor: binding.tensor(),
          file_offset_bytes: prepared_tensor.file_offset_bytes,
          bytes,
        }
        visitor(chunk)
        receipts.push({
          reference: binding.reference(),
          byte_length: binding.tensor().byte_length(),
        })
      }
      Ok({
        identity: bindings.identity(),
        tensors: ReadOnlyArray::from_array(receipts),
        total_bytes: bindings.materialized_bytes(),
      })
    }
  }
}

///|
/// Copies validated tensors once into independently owned immutable host bytes.
///
/// This path is for the bounded Phase 1 reference executor. Device loaders
/// should use `visit` so bytes move directly into their final allocation.
pub fn copy_to_host(
  bindings : @llama_weights.LlamaWeightBindings,
  file : BytesView,
  limits~ : MaterializationLimits,
) -> HostMaterialization raise MaterializationError {
  if bindings.materialized_bytes() > limits.max_host_bytes() {
    raise HostMaterializationTooLarge
  }
  let prepared = prepare(bindings, file, limits)
  let tensors : Array[HostTensor] = []
  for prepared_tensor in prepared {
    let binding = prepared_tensor.binding
    let view = prepared_tensor.bytes
    tensors.push({
      reference: binding.reference(),
      tensor: binding.tensor(),
      bytes: view.to_owned(),
    })
  }
  {
    identity: bindings.identity(),
    tensors: ReadOnlyArray::from_array(tensors),
    total_bytes: bindings.materialized_bytes(),
  }
}
