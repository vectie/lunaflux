///|
pub struct InputSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  model : @joint_diffusion_plan.JointDiffusionPlan
  sequence : @minimax_h3_sequence.Sequence
}

///|
pub fn build_input(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  sequence : @minimax_h3_sequence.Sequence,
  function_prefix~ : String,
) -> InputSource raise {
  let hidden = model.denoiser().hidden_size()
  guard sequence.assembly().width() == hidden else { raise InvalidShape }
  let shape = model.shapes()
  let width = video_projection_width(model)
  let symbols = [
    function_prefix + "_video",
    function_prefix + "_audio",
    "lunaflux_packed_rows_bf16_v1",
  ]
  let video = @decoder_projection_cuda_source.dense_projection_f32_bias_bf16_source(
    rows=sequence.assembly().input_rows(Video).to_int64(),
    input_width=width,
    output_width=hidden,
    function_symbol=symbols[0],
  ).source_bytes()
  let audio = @decoder_projection_cuda_source.dense_projection_f32_bias_bf16_source(
    rows=sequence.assembly().input_rows(Audio).to_int64(),
    input_width=shape.audio_latent_channels(),
    output_width=hidden,
    function_symbol=symbols[1],
  ).source_bytes()
  {
    source: @cuda_translation_unit.compose([
      video,
      audio,
      @packed_rows_cuda_source.render(sequence.assembly(), block_size=256),
    ]),
    symbols: ReadOnlyArray::from_array(symbols),
    model,
    sequence,
  }
}

///|
fn video_projection_width(
  model : @joint_diffusion_plan.JointDiffusionPlan,
) -> Int raise BlockSourceError {
  let shape = model.shapes()
  let mut width = shape.video_latent_channels()
  for
    d in [
      shape.video_patch_temporal(),
      shape.video_patch_height(),
      shape.video_patch_width(),
    ] {
    guard width <= 2147483647 / d else { raise InvalidShape }
    width *= d
  }
  width
}
