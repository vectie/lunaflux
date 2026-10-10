///|
/// Export the same complete AudioVAE modules used by the joint request, without
/// requiring unrelated encoder/video modules for component bring-up. Geometry
/// still comes from the resolved model request; this is not a media generator.
pub fn build_audio(
  metadata : @spec.MiniMaxH3ModelMetadata,
  plan : @loop.PreparedPlan,
  config : @config.MiniMaxH3AudioVaeConfig,
) -> SourceExport raise {
  guard @h3.build(metadata).plan_digest() == plan.model_digest() else {
    raise InvalidPlan
  }
  let units : Array[SourceUnit] = []
  append_audio(units, metadata, config, plan)
  {
    model_digest: plan.model_digest().as_hex(),
    units: ReadOnlyArray::from_array(units),
    timestep_counts: [],
  }
}

///|
fn append_audio(
  units : Array[SourceUnit],
  metadata : @spec.MiniMaxH3ModelMetadata,
  config : @config.MiniMaxH3AudioVaeConfig,
  plan : @loop.PreparedPlan,
) -> Unit raise {
  let request = plan.request()
  let transitions = @vae.bind_audio_decoder_upsample_stages(
    metadata, config, request,
  )
  let decoder = @block.build_audio_decoder(transitions[:])
  let input = @block.build_audio_input(
    time=request.audio_latent_frames(),
    max_elements=plan.audio_elements(),
  )
  units.push(
    unit("audio_input", input.source, input.symbols, [
      ("time", input.time.to_int64()),
    ]),
  )
  let front = decoder.frontend
  units.push(
    unit("audio_front", front.source, front.symbols, [
      ("time", front.time.to_int64()),
    ]),
  )
  for i, t in transitions {
    let up = decoder.upsample[i]
    units.push(
      unit("audio_up_\{i}", up.source, up.symbols, [
        ("input_time", t.input_time().to_int64()),
        ("output_time", t.output_time().to_int64()),
      ]),
    )
    let amp = decoder.amp[i]
    units.push(
      unit("audio_amp_\{i}", amp.source, amp.symbols, [
        ("channels", t.output_channels().to_int64()),
        ("time", t.output_time().to_int64()),
      ]),
    )
  }
  let tail = decoder.tail
  units.push(
    unit("audio_tail", tail.source, tail.symbols, [
      ("time", tail.time.to_int64()),
    ]),
  )
}
