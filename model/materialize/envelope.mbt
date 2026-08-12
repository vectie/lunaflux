///|
const SAFETENSORS_PREFIX_BYTES : Int = 8

///|
const MAX_INT_AS_UINT64 : UInt64 = 2147483647UL

///|
const MAX_INT_AS_INT64 : Int64 = 2147483647L

///|
priv struct ValidatedEnvelope {
  payload_start : Int
  payload_bytes : Int
}

///|
priv struct PreparedTensor {
  binding : @llama_weights.TensorBinding
  bytes : BytesView
}

///|
fn decode_header_length(file : BytesView) -> UInt64 {
  let mut length = 0UL
  for index in 0..<SAFETENSORS_PREFIX_BYTES {
    length = length | (file[index].to_uint64() << (index * 8))
  }
  length
}

///|
fn validate_envelope(
  bindings : @llama_weights.LlamaWeightBindings,
  file : BytesView,
  limits : MaterializationLimits,
) -> ValidatedEnvelope raise MaterializationError {
  if bindings.length() > limits.max_tensor_count() {
    raise TooManyTensors
  }
  if file.length() < SAFETENSORS_PREFIX_BYTES {
    raise MissingHeaderLength
  }
  let encoded_header_bytes = decode_header_length(file)
  if encoded_header_bytes > MAX_INT_AS_UINT64 {
    raise HeaderLengthOverflow
  }
  let header_bytes = encoded_header_bytes.to_int()
  if header_bytes > limits.max_header_bytes() {
    raise HeaderTooLarge
  }
  let available_after_prefix = file.length() - SAFETENSORS_PREFIX_BYTES
  if header_bytes > available_after_prefix {
    raise TruncatedHeader
  }
  let payload_start = SAFETENSORS_PREFIX_BYTES + header_bytes
  let payload_bytes = file.length() - payload_start
  let payload_bytes64 = payload_bytes.to_int64()
  if payload_bytes64 > limits.max_payload_bytes() {
    raise PayloadTooLarge
  }
  if bindings.materialized_bytes() > MAX_INT_AS_INT64 {
    raise MaterializedSizeOverflow
  }
  if bindings.materialized_bytes() != payload_bytes64 {
    raise EnvelopeLengthMismatch
  }
  { payload_start, payload_bytes }
}

///|
fn binding_at(
  bindings : @llama_weights.LlamaWeightBindings,
  index : Int,
) -> @llama_weights.TensorBinding raise MaterializationError {
  let reference = try! @plan.TensorRef::new(index)
  guard bindings.binding(reference) is Some(binding) else {
    raise MissingBinding(reference=index)
  }
  if binding.reference() != reference {
    raise BindingOrderMismatch(reference=index)
  }
  binding
}

///|
fn checked_slice(
  file : BytesView,
  envelope : ValidatedEnvelope,
  binding : @llama_weights.TensorBinding,
) -> BytesView raise MaterializationError {
  let reference = binding.reference().as_int()
  let tensor = binding.tensor()
  let start = tensor.data_start()
  let end = tensor.data_end()
  if start < 0L || end < start {
    raise InvalidTensorRange(reference~)
  }
  let payload_bytes64 = envelope.payload_bytes.to_int64()
  if end > payload_bytes64 {
    raise TensorRangeOutsidePayload(reference~)
  }
  if start > MAX_INT_AS_INT64 || end > MAX_INT_AS_INT64 {
    raise TensorRangeOverflow(reference~)
  }
  let absolute_start = envelope.payload_start + start.to_int()
  let absolute_end = envelope.payload_start + end.to_int()
  file[absolute_start:absolute_end]
}

///|
fn validate_all_ranges(
  bindings : @llama_weights.LlamaWeightBindings,
  file : BytesView,
  envelope : ValidatedEnvelope,
) -> Unit raise MaterializationError {
  let mut total = 0L
  for index in 0..<bindings.length() {
    let binding = binding_at(bindings, index)
    ignore(checked_slice(file, envelope, binding))
    let byte_length = binding.tensor().byte_length()
    if byte_length > bindings.materialized_bytes() - total {
      raise MaterializedSizeMismatch
    }
    total = total + byte_length
  }
  if total != bindings.materialized_bytes() {
    raise MaterializedSizeMismatch
  }
}

///|
fn prepare(
  bindings : @llama_weights.LlamaWeightBindings,
  file : BytesView,
  limits : MaterializationLimits,
) -> ReadOnlyArray[PreparedTensor] raise MaterializationError {
  let envelope = validate_envelope(bindings, file, limits)
  validate_all_ranges(bindings, file, envelope)
  let prepared : Array[PreparedTensor] = []
  for index in 0..<bindings.length() {
    let binding = binding_at(bindings, index)
    prepared.push({ binding, bytes: checked_slice(file, envelope, binding) })
  }
  ReadOnlyArray::from_array(prepared)
}
