///|
pub(all) enum LatentBoundary {
  PackVideoInput
  UnpackVideoOutput
  UnpackAudioOutput
}

///|
/// Immutable boundary operation, independent of device launch geometry.
/// Its input and output are distinct arena regions. No per-step permutation
/// is needed: the denoiser and RF updates retain their token-major states.
pub struct BoundaryTransfer {
  input : LatentBuffer
  output : LatentBuffer
  permutation : @dense_permutation_plan.Permutation
}

///|
pub fn LatentStoragePlan::boundary(
  self : Self,
  boundary : LatentBoundary,
) -> BoundaryTransfer {
  match boundary {
    PackVideoInput => self.video_input
    UnpackVideoOutput => self.video_output
    UnpackAudioOutput => self.audio_output
  }
}
