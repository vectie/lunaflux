///|
pub struct AudioUpsampleSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  batch : Int32
  input_channels : Int32
  output_channels : Int32
  input_time : Int32
  kernel : Int32
  stride : Int32
  padding : Int32
}

///|
pub fn build_audio_upsample(
  batch~ : Int32,
  input_channels~ : Int32,
  output_channels~ : Int32,
  input_time~ : Int32,
  kernel~ : Int32,
  stride~ : Int32,
  padding~ : Int32,
  function_prefix~ : String,
) -> AudioUpsampleSource raise {
  let symbols = [function_prefix + "_norm", function_prefix + "_conv"]
  let source = @channel_major_projection_cuda_source.f32_weight_normalized_conv_transpose1d_source(
    batch_count=batch,
    input_channels~,
    output_channels~,
    input_time~,
    kernel_size=kernel,
    padding~,
    stride~,
    normalize_symbol=symbols[0],
    convolution_symbol=symbols[1],
  ).source_bytes()
  {
    source,
    symbols: ReadOnlyArray::from_array(symbols),
    batch,
    input_channels,
    output_channels,
    input_time,
    kernel,
    stride,
    padding,
  }
}
