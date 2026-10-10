///|
pub struct AudioFrontendSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  batch : Int
  time : Int
}

///|
/// H3 decode has no encoder pre_block: pointwise dec_in_proj feeds conv_pre.
pub fn build_audio_frontend(
  batch~ : Int,
  time~ : Int,
  function_prefix~ : String,
) -> AudioFrontendSource raise {
  let symbols = [
    function_prefix + "_project",
    function_prefix + "_norm",
    function_prefix + "_conv",
  ]
  let projection = @channel_major_projection_cuda_source.f32_pointwise_conv1d_source(
    batch_count=batch,
    input_channels=32,
    output_channels=2048,
    time_count=time,
    function_symbol=symbols[0],
  )
  let convolution = @channel_major_projection_cuda_source.f32_weight_normalized_conv1d_source(
    batch_count=batch,
    input_channels=2048,
    output_channels=1024,
    input_time=time,
    kernel_size=7,
    padding=3,
    stride=1,
    normalize_symbol=symbols[1],
    convolution_symbol=symbols[2],
  )
  {
    source: @cuda_translation_unit.compose([
      projection.source_bytes(),
      convolution.source_bytes(),
    ]),
    symbols: ReadOnlyArray::from_array(symbols),
    batch,
    time,
  }
}
