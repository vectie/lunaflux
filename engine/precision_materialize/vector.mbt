///|
/// Bind a true rank-one checkpoint parameter without changing its physical
/// dtype or requiring an invented [1, N] checkpoint tensor. The execution
/// representation uses one logical row, while the source remains [N].
pub fn PackedCheckpoint::bind_vector(
  self : Self,
  representation~ : @format.Representation,
  payload_name~ : String,
) -> PackedCheckpointWeight raise PackedCheckpointError {
  let (rows, columns) = representation.shape()
  guard rows == 1 else { raise TensorStorageMismatch(payload_name) }
  let tag = match representation.format() {
    Bf16 => "BF16"
    Fp16 => "F16"
    F32 => "F32"
    _ => raise TensorStorageMismatch(payload_name)
  }
  let bytes = representation.payload_bytes()
  {
    representation,
    payload: self.checkpoint_plane(
      payload_name,
      tag,
      [columns.to_int64()],
      bytes,
    ),
    scales: None,
    global_scale: None,
    scales_offset: 0L,
    global_scale_offset: 0L,
    buffer_bytes: bytes,
  }
}
