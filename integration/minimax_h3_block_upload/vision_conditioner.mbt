///|
pub fn upload_vision_conditioner(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  packing : @vision_plan.WeightGroup,
  expected_identity~ : @spec.ModelIdentity,
  destination_offset~ : Int64,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let copies = host.resolve_vision_conditioner_weights(
    packing,
    expected_identity~,
  )
  upload_resolved(
    context,
    destination,
    host,
    copies[:],
    packing.bytes,
    destination_offset~,
    chunk_bytes~,
  )
}
