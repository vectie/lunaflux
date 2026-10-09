///|
/// Upload official upsampler or bias-free waveform-tail weights. Execution
/// programs consume this packed allocation at offset zero.
pub fn upload_audio_decoder(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  packing : @minimax_h3_weights.AudioDecoderWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let copies = host.resolve_audio_decoder_weights(packing, expected_identity~)
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
