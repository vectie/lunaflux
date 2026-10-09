///|
/// Startup-only bounded host staging into the existing device arena. Video is
/// written to its raw boundary region (pack before denoising); audio is already
/// token-major. This native RNG is not bit-equivalent to PyTorch CUDA RNG.
/// On copy failure the arena remains owned but partially initialized: retry
/// the whole operation before submission, or close it. Never submit partial
/// initialization or invoke this while frames execute.
pub fn LatentStorage::initialize_noise(
  self : Self,
  seed~ : Int64,
  chunk_elements~ : Int,
) -> Unit raise {
  let noise = self.plan.noise(seed~, max_chunk_elements=chunk_elements)
  for
    (latent, buffer) in [
      (
        @joint_diffusion_plan.JointDiffusionLatentKind::VideoLatent,
        @joint_diffusion_loop.LatentBuffer::VideoBoundary,
      ),
      (AudioLatent, AudioState),
    ] {
    let region = self.plan.region(buffer)
    let total = noise.element_count(latent)
    for start = 0L; start < total; {
      let count = if total - start < chunk_elements.to_int64() {
        (total - start).to_int()
      } else {
        chunk_elements
      }
      let bytes = noise.chunk(latent, start~, count~)
      self.allocation.copy_from_host(
        bytes,
        0L,
        region.offset + start * 4L,
        count.to_int64() * 4L,
      )
      continue start + count.to_int64()
    }
  }
}
