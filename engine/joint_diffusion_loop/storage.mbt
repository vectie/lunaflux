///|
pub(all) enum LatentBuffer {
  VideoState
  AudioState
  VideoVelocity
  AudioVelocity
  VideoBoundary
  AudioBoundary
  FlowTableBuffer
}

///|
pub(all) suberror StoragePlanError {
  ByteLimitExceeded
}

///|
pub struct StorageRegion {
  offset : Int64
  byte_count : Int64
}

///|
/// Pure layout for one request's F32 latent storage. Model weights, projected
/// hidden states, conditioning and VAE workspace are separate allocations.
/// Boundary buffers hold raw input/output only, never per-step copies.
pub struct LatentStoragePlan {
  priv regions : ReadOnlyArray[StorageRegion]
  priv bytes : Int64
  priv flow_table : FlowTable
  priv model_digest : @spec.PlanDigest
  priv request : @joint_diffusion_plan.JointDiffusionRequestShape
  priv video_input : BoundaryTransfer
  priv video_output : BoundaryTransfer
  priv audio_output : BoundaryTransfer
}

///|
pub fn LatentStoragePlan::new(
  prepared : PreparedPlan,
  max_bytes~ : Int64,
) -> Self raise StoragePlanError {
  guard max_bytes > 0L &&
    prepared.video_elements <= max_bytes / 4L &&
    prepared.audio_elements <= max_bytes / 4L else {
    raise ByteLimitExceeded
  }
  let video_bytes = prepared.video_elements * 4L
  let audio_bytes = prepared.audio_elements * 4L
  let regions : Array[StorageRegion] = []
  let bytes = for
    size in [
      video_bytes,
      audio_bytes,
      video_bytes,
      audio_bytes,
      video_bytes,
      audio_bytes,
      prepared.flow_table.byte_count(),
    ]
    cursor = 0L {
    let padding = (256L - cursor % 256L) % 256L
    guard padding <= max_bytes - cursor else { raise ByteLimitExceeded }
    let start = cursor + padding
    guard size <= max_bytes - start else { raise ByteLimitExceeded }
    regions.push({ offset: start, byte_count: size, })
    continue start + size
  } nobreak {
    cursor
  }
  {
    regions: ReadOnlyArray::from_array(regions),
    bytes,
    flow_table: prepared.flow_table,
    model_digest: prepared.model_digest,
    request: prepared.request,
    video_input: {
      input: VideoBoundary,
      output: VideoState,
      permutation: prepared.video_pack,
    },
    video_output: {
      input: VideoState,
      output: VideoBoundary,
      permutation: prepared.video_pack.inverse(),
    },
    audio_output: {
      input: AudioState,
      output: AudioBoundary,
      permutation: prepared.audio_unpack,
    },
  }
}

///|
pub fn LatentStoragePlan::byte_count(self : Self) -> Int64 {
  self.bytes
}

///|
/// Startup composition check; shape and schedule alone cannot identify a model.
pub fn LatentStoragePlan::matches_prepared(
  self : Self,
  prepared : PreparedPlan,
) -> Bool {
  self.model_digest == prepared.model_digest &&
  self.request == prepared.request &&
  self.flow_table.bytes() == prepared.flow_table.bytes()
}

///|
pub fn LatentStoragePlan::flow_table(self : Self) -> FlowTable {
  self.flow_table
}

///|
pub fn LatentStoragePlan::region(
  self : Self,
  buffer : LatentBuffer,
) -> StorageRegion {
  self.regions[match buffer {
    VideoState => 0
    AudioState => 1
    VideoVelocity => 2
    AudioVelocity => 3
    VideoBoundary => 4
    AudioBoundary => 5
    FlowTableBuffer => 6
  }]
}
