///|
/// Text and visual encoder share the official checkpoint component.
pub fn conditioner_packings() -> Array[Packing] {
  [TextEncoder, VisionConditioner]
}

///|
/// Deterministic index order: input, timestep, output, context, refiners, blocks.
pub fn denoiser_packings(
  plan : @joint.JointDiffusionPlan,
  max_bytes~ : Int64,
) -> Array[Packing] raise {
  let items : Array[Packing] = [
    Input(@weights.input_weight_packing(plan, max_bytes~)),
    Timestep(@weights.timestep_weight_packing(plan, max_bytes~)),
    Output(@weights.output_weight_packing(plan, max_bytes~)),
    Context(@weights.context_weight_packing(plan, max_bytes~)),
  ]
  for layer in 0..<plan.denoiser().refiner_layer_count() {
    items.push(
      Refiner(@weights.refiner_weight_packing(plan, layer~, max_bytes~)),
    )
  }
  for block in 0..<plan.denoiser().layer_count() {
    items.push(Block(@weights.block_weight_packing(plan, block~, max_bytes~)))
  }
  component_bytes(Fl2vaDenoiser, items[:], max_bytes) |> ignore
  items
}

///|
/// Official VideoVAE decoder: frontend, 36 transformer blocks, final head.
pub fn video_decoder_packings(max_bytes~ : Int64) -> Array[Packing] raise {
  let items : Array[Packing] = [
    VideoFrontend(@weights.video_frontend_weight_packing(max_bytes~)),
  ]
  for layer in 0..<36 {
    items.push(
      VideoBlock(@weights.video_block_weight_packing(layer~, max_bytes~)),
    )
  }
  items.push(VideoFrontend(@weights.video_head_weight_packing(max_bytes~)))
  component_bytes(VideoVae, items[:], max_bytes) |> ignore
  items
}

///|
/// Official AudioVAE decoder: frontend, seven (upsample,AMP) pairs, tail.
/// Startup normalization caches are owned separately by AudioDecoder and are
/// included in that program's explicit cache budget, not these raw weights.
pub fn audio_decoder_packings(max_bytes~ : Int64) -> Array[Packing] raise {
  let items : Array[Packing] = [
    AudioDecoder(@weights.audio_decoder_weight_packing(Frontend, max_bytes~)),
  ]
  for stage in 0..<7 {
    items.push(
      AudioDecoder(
        @weights.audio_decoder_weight_packing(Upsample(stage), max_bytes~),
      ),
    )
    items.push(
      AudioAmp(@vae.bind_audio_decoder_amp_parameter_layout(stage_index=stage)),
    )
  }
  items.push(
    AudioDecoder(
      @weights.audio_decoder_weight_packing(WaveformTail, max_bytes~),
    ),
  )
  component_bytes(AudioVae, items[:], max_bytes) |> ignore
  items
}
