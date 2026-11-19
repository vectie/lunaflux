///|
pub(all) suberror TransferError {
  BindingMismatch
  InvalidChunk
}

///|
pub struct WeightTransfer {
  ordinal : Int32
  source_offset : Int64
  source_bytes : Int64
  destination : @layout.Region
  expanded_norm : @layout.Region?
}

///|
/// Join the existing LunaFlux Qwen safetensors binder to the FPGA placement.
/// Source offsets are relative to the safetensors data section, not the file.
/// No weight bytes are read, hashed, copied, or materialized by this function.
pub fn prepare_transfers(
  layout : @layout.Layout,
  bindings : @weights.Qwen3WeightBindings,
) -> ReadOnlyArray[WeightTransfer] raise TransferError {
  if bindings.content_digest() != layout.content_label() ||
    bindings.length() != layout.weight_count() {
    raise BindingMismatch
  }
  let output : Array[WeightTransfer] = []
  for ordinal in 0..<layout.weight_count() {
    let reference = @plan.TensorRef::new(ordinal) catch {
      _ => raise BindingMismatch
    }
    guard bindings.binding(reference) is Some(binding) &&
      layout.weight(ordinal) is Some(destination) &&
      layout.dimensions(ordinal) is Some(dimensions) else {
      raise BindingMismatch
    }
    let tensor = binding.tensor()
    if tensor.dtype() != BFloat16 ||
      tensor.shape() != dimensions ||
      tensor.byte_length() < 0L ||
      tensor.byte_length().reinterpret_as_uint64() != destination.bytes {
      raise BindingMismatch
    }
    output.push({
      ordinal,
      source_offset: tensor.data_start(),
      source_bytes: tensor.byte_length(),
      destination,
      expanded_norm: layout.expanded_norm(ordinal),
    })
  }
  ReadOnlyArray::from_array(output)
}

///|
pub(all) struct TransferChunk {
  source_offset : Int64
  source_bytes : Int32
  destination_address : UInt64
}

///|
/// Bounded startup streaming. Use one reusable host buffer; no full-checkpoint
/// intermediate is necessary. The tied output head has no separate transfer.
pub fn WeightTransfer::chunk(
  self : Self,
  offset : Int64,
  max_bytes : Int32,
) -> TransferChunk? raise TransferError {
  if offset < 0L ||
    offset > self.source_bytes ||
    offset % 2L != 0L ||
    max_bytes < 2 ||
    max_bytes > 1048576 ||
    max_bytes % 2 != 0 {
    raise InvalidChunk
  }
  if offset == self.source_bytes {
    return None
  }
  let count = (self.source_bytes - offset).min(max_bytes.to_int64()).to_int()
  Some({
    source_offset: self.source_offset + offset,
    source_bytes: count,
    destination_address: self.destination.address +
    offset.reinterpret_as_uint64(),
  })
}

///|
/// Exact BF16 -> FP32 expansion into a caller-owned buffer. Used for small
/// normalization vectors (and a future device-read embedding row), never as
/// a request to duplicate the full embedding/output matrix.
pub fn expand_bf16(
  source : BytesView,
  destination : Array[Byte],
) -> Unit raise TransferError {
  if source.is_empty() ||
    source.length() % 2 != 0 ||
    source.length() > 8192 ||
    destination.length() != source.length() * 2 {
    raise InvalidChunk
  }
  for i in 0..<(source.length() / 2) {
    destination[i * 4] = b'\x00'
    destination[i * 4 + 1] = b'\x00'
    destination[i * 4 + 2] = source[i * 2]
    destination[i * 4 + 3] = source[i * 2 + 1]
  }
}
