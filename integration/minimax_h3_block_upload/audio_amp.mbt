///|
/// Pack admitted F32 AudioVAE AMP parameters into the exact execution layout.
/// Upload at offset zero when passing this allocation to AudioAmpProgram.bind.
pub fn upload_audio_amp(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  parameters : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  expected_identity~ : @spec.ModelIdentity,
  destination_offset~ : Int64,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let copies = host.resolve_audio_amp_weights(parameters, expected_identity~)
  upload_resolved(
    context,
    destination,
    host,
    copies[:],
    parameters.total_arena_bytes(),
    destination_offset~,
    chunk_bytes~,
  )
}
