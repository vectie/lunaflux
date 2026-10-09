///|
pub struct AudioAmpSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  batch : Int
  channels : Int
  time : Int
}

///|
/// Offline bundle for AudioAmpLayout's 41-function ABI. The three parallel
/// branches use kernels 3/7/11 and residual-pair dilations 1/3/5, then 1.
pub fn build_audio_amp(
  batch~ : Int,
  channels~ : Int,
  time~ : Int,
  function_prefix~ : String,
) -> AudioAmpSource raise {
  let symbols = [
    function_prefix + "_up",
    function_prefix + "_snake",
    function_prefix + "_down",
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
  let mut source = activation.source_bytes()
  for block, kernel in [3, 7, 11] {
    for pair, first_dilation in [1, 3, 5] {
      for side in 0..<2 {
        let dilation = if side == 0 { first_dilation } else { 1 }
        let prefix = "\{function_prefix}_b\{block}_p\{pair}_s\{side}"
        let normalize = prefix + "_norm"
        let convolve = prefix + "_conv"
        symbols.push(normalize)
        symbols.push(convolve)
        source = source +
          @channel_major_projection_cuda_source.f32_weight_normalized_dilated_conv1d_source(
            batch_count=batch,
            channel_count=channels,
            time_count=time,
            kernel_size=kernel,
            dilation~,
            padding=(kernel - 1) * dilation / 2,
            normalize_symbol=normalize,
            convolution_symbol=convolve,
          ).source_bytes()
      }
    }
  }
  let residual = function_prefix + "_residual"
  let average = function_prefix + "_average"
  symbols.push(residual)
  symbols.push(average)
  source = source +
    @joint_audio_amp_cuda_source.f32_residual_and_three_way_average_source(
      element_count=activation.input_element_count(),
      residual_symbol=residual,
      average_symbol=average,
    ).source_bytes()
  {
    source,
    symbols: ReadOnlyArray::from_array(symbols),
    batch,
    channels,
    time,
  }
}
