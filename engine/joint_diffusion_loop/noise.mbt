///|
pub(all) suberror NoiseError {
  InvalidChunkLimit
  InvalidRange
}

///|
/// Stateless native startup generator. Version 1 follows the independent
/// SplitMix64/Box-Muller reference, not PyTorch's seed-to-tensor sequence.
/// Video counters enumerate raw channel-major cells; audio counters enumerate
/// stream-major token rows. Chunk boundaries do not affect any sample.
pub struct NoisePlan {
  priv key : UInt64
  priv video_elements : Int64
  priv audio_elements : Int64
  priv chunk_elements : Int
}

///|
pub fn LatentStoragePlan::noise(
  self : Self,
  seed~ : Int64,
  max_chunk_elements~ : Int,
) -> NoisePlan raise NoiseError {
  guard max_chunk_elements > 0 && max_chunk_elements <= 536870911 else {
    raise InvalidChunkLimit
  }
  let digest = self.model_digest.as_hex()
  let hash = for index in 0..<digest.length(); hash = 14695981039346656037UL {
    continue (hash ^ digest[index].to_uint64()) * 1099511628211UL
  } nobreak {
    hash
  }
  {
    key: seed.reinterpret_as_uint64() ^ hash,
    video_elements: self.region(VideoState).byte_count / 4L,
    audio_elements: self.region(AudioState).byte_count / 4L,
    chunk_elements: max_chunk_elements,
  }
}

///|
fn noise_mix(input : UInt64) -> UInt64 {
  let first = input + 0x9e3779b97f4a7c15UL
  let second = (first ^ (first >> 30)) * 0xbf58476d1ce4e5b9UL
  let third = (second ^ (second >> 27)) * 0x94d049bb133111ebUL
  third ^ (third >> 31)
}

///|
pub fn NoisePlan::element_count(
  self : Self,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int64 {
  match latent {
    VideoLatent => self.video_elements
    AudioLatent => self.audio_elements
  }
}

///|
/// Generate only the requested bounded chunk as little-endian F32. Bounds are
/// checked once, outside the numerical loop. No global RNG state or reference
/// package dependency, no tensor-sized host temporary and no step-path work.
pub fn NoisePlan::chunk(
  self : Self,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
  start~ : Int64,
  count~ : Int,
) -> Bytes raise NoiseError {
  let limit = self.element_count(latent)
  guard start >= 0L &&
    start <= limit &&
    count > 0 &&
    count <= self.chunk_elements &&
    count.to_int64() <= limit - start else {
    raise InvalidRange
  }
  let domain = match latent {
    VideoLatent => 0x13198a2e03707344UL
    AudioLatent => 0xa4093822299f31d0UL
  }
  let output = FixedArray::make(count * 4, b'\x00')
  for index in 0..<count {
    let counter = (start + index.to_int64()).reinterpret_as_uint64()
    let base = self.key ^ domain ^ (counter * 0x9e3779b97f4a7c15UL)
    let first = (
        (noise_mix(base ^ 0xd1b54a32d192ed03UL) >> 11).to_double() + 0.5
      ) /
      9007199254740992.0
    let second = (
        (noise_mix(base ^ 0x94d049bb133111ebUL) >> 11).to_double() + 0.5
      ) /
      9007199254740992.0
    let value = (-2.0 * @math.ln(first)).sqrt() *
      @math.cos(2.0 * @math.PI * second)
    write_f32(output, index * 4, Float::from_double(value))
  }
  Bytes::of(output)
}
