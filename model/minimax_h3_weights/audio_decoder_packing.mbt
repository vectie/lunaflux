///|
pub(all) enum AudioDecoderWeightGroup {
  Frontend
  Upsample(Int)
  WaveformTail
} derive(Eq)

///|
pub struct AudioDecoderWeightPacking {
  group : AudioDecoderWeightGroup
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
/// Released AudioVAE F32 weights in execution ABI order. Tail deliberately
/// omits conv_post.bias: official H3 disables that bias and final tanh.
pub fn audio_decoder_weight_packing(
  group : AudioDecoderWeightGroup,
  max_bytes~ : Int64,
) -> AudioDecoderWeightPacking raise BlockPackingError {
  let (names, shapes) : (Array[String], Array[ReadOnlyArray[Int64]]) = match
    group {
    Frontend =>
      (
        [
          "dec_in_proj.weight", "dec_in_proj.bias", "decoder.conv_pre.weight_g",
          "decoder.conv_pre.weight_v", "decoder.conv_pre.bias",
        ],
        [
          [2048L, 32L, 1L],
          [2048L],
          [1024L, 1L, 1L],
          [1024L, 2048L, 7L],
          [1024L],
        ],
      )
    Upsample(stage) => {
      guard stage >= 0 && stage < 7 else { raise BudgetExceeded }
      let ci = 1024L >> stage
      let co = ci / 2L
      let kernel = if stage < 2 { 9L } else { 4L }
      let prefix = "decoder.ups.\{stage}.0"
      (
        [prefix + ".weight_g", prefix + ".weight_v", prefix + ".bias"],
        [[ci, 1L, 1L], [ci, co, kernel], [co]],
      )
    }
    WaveformTail =>
      (
        [
          "decoder.activation_post.act.alpha", "decoder.activation_post.act.beta",
          "decoder.activation_post.upsample.filter", "decoder.activation_post.downsample.lowpass.filter",
          "decoder.conv_post.weight_g", "decoder.conv_post.weight_v",
        ],
        [[8L], [8L], [1L, 1L, 12L], [1L, 1L, 12L], [1L, 1L, 1L], [1L, 8L, 7L]],
      )
  }
  let schema = MiniMaxH3AudioVaeSchema::official_v1()
  let copies : Array[BlockWeightCopy] = []
  let bytes = for i, name in names; offset = 0L {
    guard schema.matches_tensor(name, Float32, shapes[i]) else {
      raise BudgetExceeded
    }
    let size = shapes[i].fold(init=4L, (bytes, extent) => bytes * extent)
    guard size <= max_bytes - offset else { raise BudgetExceeded }
    copies.push({
      tensor_name: name,
      source_offset: 0L,
      destination_offset: offset,
      byte_count: size,
    })
    continue offset + size
  } nobreak {
    offset
  }
  { group, byte_count: bytes, copies: ReadOnlyArray::from_array(copies), }
}
