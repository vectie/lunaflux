///|
pub struct OutputSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  video_start : Int
  video_rows : Int
  audio_start : Int
  audio_rows : Int
  model : @joint_diffusion_plan.JointDiffusionPlan
  packed_rows : Int
  distinct_timesteps : Int
  packed_mapping : Bytes
}

///|
/// Final parameter projection consumes the same preactivated BF16 timestep
/// rows as the blocks, but has one group and two fields. Modulation indices
/// must therefore be timestep indices, not the three-modality block indices.
pub fn build_output(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  sequence : @minimax_h3_sequence.Sequence,
  distinct_timesteps~ : Int,
  function_prefix~ : String,
) -> OutputSource raise {
  let shape = model.denoiser()
  let hidden = shape.hidden_size()
  guard sequence.assembly().width() == hidden && distinct_timesteps > 0 else {
    raise InvalidShape
  }
  let mut video_start = 0
  let mut audio_start = 0
  let mut video_rows = 0
  let mut audio_rows = 0
  for segment in sequence.segments() {
    match segment.kind {
      TargetVideo => {
        video_start = segment.start
        video_rows = segment.count
      }
      TargetAudio => {
        audio_start = segment.start
        audio_rows = segment.count
      }
      _ => ()
    }
  }
  guard video_rows > 0 && audio_rows > 0 else { raise InvalidShape }
  let geometry = model.shapes()
  let mut video_width = 1
  for
    dimension in [
      geometry.video_latent_channels(),
      geometry.video_patch_temporal(),
      geometry.video_patch_height(),
      geometry.video_patch_width(),
    ] {
    guard dimension > 0 && video_width <= 2147483647 / dimension else {
      raise InvalidShape
    }
    video_width *= dimension
  }
  let symbols = [
    function_prefix + "_parameters",
    function_prefix + "_norm",
    function_prefix + "_video_mod",
    function_prefix + "_audio_mod",
    function_prefix + "_video_out",
    function_prefix + "_audio_out",
  ]
  let parameters = @affine_modulation_cuda_source.grouped_parameter_projection_source(
    rows=distinct_timesteps,
    input_width=shape.final_adaln_conditioning_dimension(),
    width=hidden,
    groups=1,
    fields=2,
    function_symbol=symbols[0],
  )
  let norm = @decoder_foundation_cuda_source.rms_norm_bf16_fixed_rows_source(
    rows=sequence.assembly().rows().length().to_int64(),
    hidden_size=hidden,
    epsilon=shape.final_norm_epsilon(),
    function_symbol=symbols[1],
  ).source_bytes()
  let video_mod = @affine_modulation_cuda_source.indexed_bf16_shift_scale_f32_source(
    rows=video_rows,
    width=hidden,
    parameter_rows=distinct_timesteps,
    function_symbol=symbols[2],
  )
  let audio_mod = @affine_modulation_cuda_source.indexed_bf16_shift_scale_f32_source(
    rows=audio_rows,
    width=hidden,
    parameter_rows=distinct_timesteps,
    function_symbol=symbols[3],
  )
  let video = @decoder_projection_cuda_source.dense_projection_f32_bias_f32_source(
    rows=video_rows.to_int64(),
    input_width=hidden,
    output_width=video_width,
    function_symbol=symbols[4],
  ).source_bytes()
  let audio = @decoder_projection_cuda_source.dense_projection_f32_bias_f32_source(
    rows=audio_rows.to_int64(),
    input_width=hidden,
    output_width=geometry.audio_latent_channels(),
    function_symbol=symbols[5],
  ).source_bytes()
  {
    source: @cuda_translation_unit.compose([
      parameters, norm, video_mod, audio_mod, video, audio,
    ]),
    symbols: ReadOnlyArray::from_array(symbols),
    video_start,
    video_rows,
    audio_start,
    audio_rows,
    model,
    packed_rows: sequence.assembly().rows().length(),
    distinct_timesteps,
    packed_mapping: sequence.assembly().mapping(),
  }
}
