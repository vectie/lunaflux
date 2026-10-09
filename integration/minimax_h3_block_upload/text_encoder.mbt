///|
/// Startup upload into one caller-owned layer allocation, at byte offset zero.
pub fn upload_text_encoder_layer(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  packing : @text_plan.LayerWeights,
  expected_identity~ : @spec.ModelIdentity,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let copies = host.resolve_text_encoder_layer(packing, expected_identity~)
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

///|
pub fn upload_text_encoder_embedding(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  expected_identity~ : @spec.ModelIdentity,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let copies = host.resolve_text_encoder_embedding(expected_identity~)
  upload_resolved(
    context,
    destination,
    host,
    copies[:],
    @text_plan.embedding_region().bytes,
    destination_offset=0L,
    chunk_bytes~,
  )
}
