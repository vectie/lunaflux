///|
pub struct ConditioningCopy {
  rows : Int
  destination_offset : Int64
  byte_count : Int64
  latent : @joint_diffusion_plan.JointDiffusionLatentKind?
  input_width : Int
}

///|
pub struct ConditioningSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  copies : ReadOnlyArray[ConditioningCopy]
  model : @joint_diffusion_plan.JointDiffusionPlan
  sequence : @minimax_h3_sequence.Sequence
  byte_count : Int64
}

///|
/// Text followed by projected reference blocks, in the same order as packed
/// assembly. Target latents and padding are never copied into conditioning.
pub fn build_conditioning(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  sequence : @minimax_h3_sequence.Sequence,
  function_prefix~ : String,
) -> ConditioningSource raise {
  let width = model.denoiser().hidden_size()
  guard sequence.assembly().width() == width else { raise InvalidShape }
  let copies : Array[ConditioningCopy] = []
  let symbols : Array[String] = []
  let fragments : Array[Bytes] = []
  let mut bytes = 0L
  let projections : Array[(String, Bytes)] = []
  for segment in sequence.segments() {
    match segment.kind {
      TargetAudio | TargetVideo => ()
      Text | ConditionAudio | ConditionVideo => {
        let elements = segment.count.to_int64() * width.to_int64()
        let symbol = function_prefix + "_copy_" + copies.length().to_string()
        fragments.push(
          @decoder_foundation_cuda_source.copy_bf16_source(
            elements~,
            function_symbol=symbol,
          ),
        )
        symbols.push(symbol)
        let latent = match segment.kind {
          ConditionAudio =>
            Some(@joint_diffusion_plan.JointDiffusionLatentKind::AudioLatent)
          ConditionVideo =>
            Some(@joint_diffusion_plan.JointDiffusionLatentKind::VideoLatent)
          _ => None
        }
        let input_width = match latent {
          Some(AudioLatent) => model.shapes().audio_latent_channels()
          Some(VideoLatent) => video_projection_width(model)
          None => width
        }
        if latent is Some(_) {
          let projection_symbol = function_prefix +
            "_project_" +
            copies.length().to_string()
          projections.push(
            (
              projection_symbol,
              @decoder_projection_cuda_source.dense_projection_f32_bias_bf16_source(
                rows=segment.count.to_int64(),
                input_width~,
                output_width=width,
                function_symbol=projection_symbol,
              ).source_bytes(),
            ),
          )
        }
        copies.push({
          rows: segment.count,
          destination_offset: bytes,
          byte_count: elements * 2L,
          latent,
          input_width,
        })
        bytes += elements * 2L
      }
    }
  }
  for (symbol, projection) in projections {
    symbols.push(symbol)
    fragments.push(projection)
  }
  {
    source: @cuda_translation_unit.compose(fragments[:]),
    symbols: ReadOnlyArray::from_array(symbols),
    copies: ReadOnlyArray::from_array(copies),
    model,
    sequence,
    byte_count: bytes,
  }
}
