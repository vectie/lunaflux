///|
/// Upload pure checkpoint slice plans through bounded reusable staging. Device
/// destinations remain owned by the caller on success and partial failure.
pub fn upload_streamed(
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  component : @minimax_h3_host_materialization.MiniMaxH3StreamingComponent,
  allocations : ArrayView[@device.Allocation],
  requests : ArrayView[@streaming_safetensors.StreamingSafetensorsTransfer],
  destination_bytes~ : ArrayView[Int64],
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 && allocations.length() == destination_bytes.length() else {
    raise InvalidChunkSize
  }
  for i, allocation in allocations {
    context.validate_allocation_region(
      allocation,
      offset=0L,
      byte_count=destination_bytes[i],
      alignment=2L,
    )
  }
  let scratch = FixedArray::make(chunk_bytes, b'\x00')
  component.transfer(root, requests, destination_bytes~, scratch~, consume=(
    destination,
    offset,
    buffer,
    count,
  ) => {
    let bytes = Bytes::from_array(buffer[:count])
    try
      allocations[destination].copy_from_host(
        bytes,
        0L,
        offset,
        count.to_int64(),
      )
    catch {
      _ => false
    } noraise {
      _ => true
    }
  })
}
