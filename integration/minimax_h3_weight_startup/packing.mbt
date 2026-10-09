///|
/// Pure packing descriptors produced by the model-owned packers. Order is
/// preserved in the resulting allocation indices; no runtime tensor guessing.
pub(all) enum Packing {
  TextEncoder
  VisionConditioner
  ReferenceEncoder(@segmented.SegmentedDeviceMaterializationLimits, Int64)
  Block(@weights.BlockWeightPacking)
  Input(@weights.InputWeightPacking)
  Timestep(@weights.TimestepWeightPacking)
  Output(@weights.OutputWeightPacking)
  Context(@weights.ContextWeightPacking)
  Refiner(@weights.RefinerWeightPacking)
  VideoFrontend(@weights.VideoFrontendWeightPacking)
  VideoBlock(@weights.VideoBlockWeightPacking)
  AudioDecoder(@weights.AudioDecoderWeightPacking)
  AudioAmp(@amp.JointAudioAmpParameterLayout)
}

///|
pub(all) suberror StartupError {
  InvalidState
  InvalidPacking
  BudgetExceeded
}

///|
pub fn Packing::bytes(self : Self) -> Int64 raise StartupError {
  try {
    match self {
      TextEncoder =>
        @text_plan.embedding_region().bytes +
        50L * @text_plan.layer_weights(0).byte_count
      VisionConditioner => {
        let mut n = @vision_plan.frontend_weights().bytes
        for i in 0..<27 {
          n += @vision_plan.block_weights(i).bytes
        }
        for i in 0..<4 {
          n += @vision_plan.merger_weights(i).bytes
        }
        n
      }
      Block(p) => p.byte_count
      Input(p) => p.byte_count
      Timestep(p) => p.byte_count
      Output(p) => p.byte_count
      Context(p) => p.byte_count
      Refiner(p) => p.byte_count
      VideoFrontend(p) => p.byte_count
      VideoBlock(p) => p.byte_count
      AudioDecoder(p) => p.byte_count
      AudioAmp(p) => p.total_arena_bytes()
      ReferenceEncoder(_, reservation) => reservation
    }
  } catch {
    _ => raise InvalidPacking
  }
}

///|
fn accepts(
  component : @weights.MiniMaxH3WeightComponent,
  packing : Packing,
) -> Bool {
  match packing {
    TextEncoder | VisionConditioner => component == TextConditioner
    VideoFrontend(_) | VideoBlock(_) => component == VideoVae
    AudioDecoder(_) | AudioAmp(_) => component == AudioVae
    ReferenceEncoder(_, _) => component == AudioVae || component == VideoVae
    _ => component == Fl2vaDenoiser || component == Ref2vaDenoiser
  }
}

///|
/// Subtraction-first bounded accounting cannot overflow even with an Int64
/// maximum budget. Runs before materialization or the first allocation.
pub fn component_bytes(
  component : @weights.MiniMaxH3WeightComponent,
  packings : ArrayView[Packing],
  remaining : Int64,
) -> Int64 raise StartupError {
  guard remaining > 0L && !packings.is_empty() else { raise BudgetExceeded }
  let mut left = remaining
  for packing in packings {
    guard accepts(component, packing) else { raise InvalidPacking }
    let bytes = packing.bytes()
    guard bytes > 0L && bytes <= left else { raise BudgetExceeded }
    left -= bytes
  }
  remaining - left
}
