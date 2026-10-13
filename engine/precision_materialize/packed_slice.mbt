///|
/// Bind a compiler-owned packed view to an already checked source tensor.
/// Source metadata remains the original metadata. Upload reads only the
/// selected ranges; scale grids and scalar global scales are not recomputed.
pub fn PackedCheckpointWeight::slice(
  self : Self,
  view : @precision.PackedMatrixSlice,
) -> Self raise PackedCheckpointError {
  guard self.slice is None && self.representation == view.source() else {
    raise TensorStorageMismatch(self.payload.metadata.name())
  }
  let layout = @precision.packed_buffer_layout(view.representation()) catch {
    _ => raise BufferSizeOverflow
  }
  {
    ..self,
    representation: view.representation(),
    scales_offset: layout.scales_offset().unwrap_or(0L),
    global_scale_offset: layout.global_scale_offset().unwrap_or(0L),
    buffer_bytes: layout.buffer_bytes(),
    slice: Some(view),
  }
}

///|
fn append_plane_transfers(
  requests : Array[@streaming.StreamingSafetensorsTransfer],
  location : CheckpointLocation,
  destination : Int,
  offset : Int64,
  view~ : @precision.PackedPlaneSlice?,
) -> Unit raise PackedCheckpointError {
  guard view is Some(view) else {
    requests.push(transfer_plane(location, destination, offset))
    return
  }
  let (source, stride, rows, bytes) = view.geometry()
  // One descriptor even for a column shard. The streaming reader compacts
  // bounded sequential windows; no per-row descriptor array or full tensor.
  let request = @streaming.StreamingSafetensorsTransfer::new(
    shard_ordinal=location.shard,
    tensor_ordinal=location.tensor,
    tensor_byte_offset=source,
    destination_ordinal=destination,
    destination_offset=offset,
    byte_count=rows.to_int64() * bytes,
    row_count=rows,
    source_row_stride=stride,
  ) catch {
    error => raise Streaming(error)
  }
  requests.push(request)
}
