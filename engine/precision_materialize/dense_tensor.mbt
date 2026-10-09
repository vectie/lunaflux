///|
/// Bind an explicitly shaped contiguous dense tensor to an execution matrix
/// view. No transpose, dtype conversion, packing change or data copy occurs.
/// The caller supplies the checkpoint shape; equal element counts alone do
/// not permit substituting another physical shape.
pub fn PackedCheckpoint::bind_dense_tensor(
  self : Self,
  representation~ : @format.Representation,
  payload_name~ : String,
  source_shape~ : ArrayView[Int64],
) -> PackedCheckpointWeight raise PackedCheckpointError {
  let tag = match representation.format() {
    Bf16 => "BF16"
    Fp16 => "F16"
    F32 => "F32"
    _ => raise TensorStorageMismatch(payload_name)
  }
  let (rows, columns) = representation.shape()
  let cells = for dimension in source_shape; product = 1L {
    guard dimension > 0L && product <= 9223372036854775807L / dimension else {
      raise TensorStorageMismatch(payload_name)
    }
    continue product * dimension
  } nobreak {
    product
  }
  guard cells == rows.to_int64() * columns.to_int64() else {
    raise TensorStorageMismatch(payload_name)
  }
  let bytes = representation.payload_bytes()
  {
    representation,
    payload: self.checkpoint_plane(payload_name, tag, source_shape, bytes),
    scales: None,
    global_scale: None,
    scales_offset: 0L,
    global_scale_offset: 0L,
    buffer_bytes: bytes,
  }
}
