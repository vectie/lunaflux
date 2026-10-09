///|
pub(all) suberror FlowTableError {
  StepOutOfBounds
}

///|
/// Immutable little-endian F32 startup payload. Each step occupies 32 bytes:
/// video [sigma_t, ratio, complement, timestep], then the same audio record.
/// The RF kernel binds only the first 12 bytes of each 16-byte record.
pub struct FlowTable {
  priv payload : Bytes
  priv steps : Int
}

///|
fn write_f32(output : FixedArray[Byte], offset : Int, value : Float) -> Unit {
  let bits = value.reinterpret_as_uint()
  for byte in 0..<4 {
    output[offset + byte] = (bits >> (byte * 8)).to_byte()
  }
}

///|
fn build_flow_table(steps : ReadOnlyArray[JointStep]) -> FlowTable {
  // PreparedPlan admits at most the shape contract's one million steps.
  let output = FixedArray::make(steps.length() * 32, b'\x00')
  for index, step in steps {
    let video = step.video_coefficients()
    let audio = step.audio_coefficients()
    let base = index * 32
    write_f32(output, base, video.sigma_t)
    write_f32(output, base + 4, video.ratio)
    write_f32(output, base + 8, video.complement)
    write_f32(output, base + 12, Float::from_double(step.video_timestep))
    write_f32(output, base + 16, audio.sigma_t)
    write_f32(output, base + 20, audio.ratio)
    write_f32(output, base + 24, audio.complement)
    write_f32(output, base + 28, Float::from_double(step.audio_timestep))
  }
  { payload: Bytes::of(output), steps: steps.length(), }
}

///|
pub fn FlowTable::bytes(self : Self) -> Bytes {
  self.payload
}

///|
pub fn FlowTable::byte_count(self : Self) -> Int64 {
  self.payload.length().to_int64()
}

///|
/// Startup binding offset, never computed by the device kernel.
pub fn FlowTable::coefficient_offset(
  self : Self,
  index : Int,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int64 raise FlowTableError {
  guard index >= 0 && index < self.steps else { raise StepOutOfBounds }
  let modality_offset = match latent {
    VideoLatent => 0L
    AudioLatent => 16L
  }
  index.to_int64() * 32L + modality_offset
}
