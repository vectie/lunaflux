///|
pub fn upload_video_block(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  packing : @minimax_h3_weights.VideoBlockWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let copies = host.resolve_video_block_weights(packing, expected_identity~)
  upload_resolved(
    context,
    destination,
    host,
    copies[:],
    packing.byte_count,
    destination_offset=0L,
    chunk_bytes~,
  )
}
