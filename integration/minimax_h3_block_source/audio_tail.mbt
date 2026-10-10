///|
pub struct AudioTailSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  batch : Int
  channels : Int
  time : Int
}

///|
/// BigVGAN waveform tail: alias-free SnakeBeta, 7-tap channel-to-one
/// bias-free convolution. Official H3 disables final tanh as well as bias.
pub fn build_audio_tail(
  batch~ : Int,
  channels~ : Int,
  time~ : Int,
  function_prefix~ : String,
) -> AudioTailSource raise {
  let symbols = [
    function_prefix + "_up",
    function_prefix + "_snake",
    function_prefix + "_down",
    function_prefix + "_norm",
    function_prefix + "_conv",
  ]
  let activation = @alias_free_activation_cuda_source.f32_alias_free_snake_beta_source(
    batch_count=batch,
    channel_count=channels,
    time_count=time,
    ratio=2,
    kernel_size=12,
    upsample_symbol=symbols[0],
    activation_symbol=symbols[1],
    downsample_symbol=symbols[2],
  )
  let convolution = @channel_major_projection_cuda_source.f32_weight_normalized_conv1d_source(
    batch_count=batch,
    input_channels=channels,
    output_channels=1,
    input_time=time,
    kernel_size=7,
    padding=3,
    stride=1,
    normalize_symbol=symbols[3],
    convolution_symbol=symbols[4],
    use_bias=false,
  )
  {
    source: @cuda_translation_unit.compose([
      activation.source_bytes(),
      convolution.source_bytes(),
    ]),
    symbols: ReadOnlyArray::from_array(symbols),
    batch,
    channels,
    time,
  }
}
