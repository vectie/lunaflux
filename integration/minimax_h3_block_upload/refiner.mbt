///|
pub fn upload_refiner(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  packing : @minimax_h3_weights.RefinerWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
  destination_offset~ : Int64,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let copies = host.resolve_refiner_weights(packing, expected_identity~)
  upload_resolved(
    context,
    destination,
    host,
    copies[:],
    packing.byte_count,
    destination_offset~,
    chunk_bytes~,
  )
}
