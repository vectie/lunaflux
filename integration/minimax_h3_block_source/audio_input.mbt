///|
pub struct AudioInputSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  time : Int
  value_bytes : Int64
}

///|
/// One module per resolved shape: the shared permutation renderer exports its
/// stable ABI symbol. Statistics are runtime immutable operands, not constants.
pub fn build_audio_input(
  time~ : Int,
  max_elements~ : Int64,
) -> AudioInputSource raise {
  let permutation = @dense_permutation_plan.Permutation::new(
    [2, time, 32],
    [0, 2, 1],
    max_elements~,
  )
  let affine = @affine_modulation_cuda_source.f32_channel_major_destandardization_source(
    batch_count=2,
    channels=32,
    elements_per_channel=time.to_int64(),
    function_symbol="lunaflux_audio_destandardize_v1",
  )
  {
    source: @cuda_translation_unit.compose([
      @dense_permutation_cuda_source.render(permutation, block_size=256),
      affine.source_bytes(),
    ]),
    symbols: [
      "lunaflux_dense_permutation_u32_v1", "lunaflux_audio_destandardize_v1",
    ],
    time,
    value_bytes: permutation.element_count() * 4L,
  }
}
