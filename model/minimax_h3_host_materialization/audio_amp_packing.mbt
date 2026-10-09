///|
fn resolve_audio_amp_regions(
  tensors : ArrayView[MiniMaxH3HostTensor],
  parameters : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  let copies = parameters
    .regions()
    .map(region => {
      let name = region.tensor_name()
      let resolved = resolve_audio_whole_tensor(
        tensors,
        name,
        region.byte_count(),
        region.byte_offset(),
      )
      resolved
    })
  ReadOnlyArray::from_array(copies)
}

///|
/// Startup-only whole-tensor copies from the admitted AudioVAE component.
/// The same parameter layout is passed to AudioAmpProgram.bind.
pub fn MiniMaxH3HostComponentArena::resolve_audio_amp_weights(
  self : Self,
  parameters : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  guard !self.released else { raise Unavailable }
  guard self.identity == expected_identity else { raise IdentityMismatch }
  guard self.component is AudioVae else { raise InvalidComponent }
  resolve_audio_amp_regions(self.tensors, parameters)
}
