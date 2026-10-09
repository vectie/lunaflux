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
  let input = @block.build_audio_input(
    time=request.audio_latent_frames(),
    max_elements=plan.audio_elements(),
  )
  units.push(
    unit("audio_input", input.source, input.symbols, [
      ("time", input.time.to_int64()),
    ]),
  )
  let front = @block.build_audio_frontend(
    batch=2,
    time=request.audio_latent_frames(),
    function_prefix="h3_audio_front",
  )
  units.push(
    unit("audio_front", front.source, front.symbols, [
      ("time", front.time.to_int64()),
    ]),
  )
  for i, t in transitions {
    let up = @block.build_audio_upsample(
      batch=t.decoder_batch_count(),
      input_channels=t.input_channels(),
      output_channels=t.output_channels(),
      input_time=t.input_time(),
      kernel=t.kernel_size(),
      stride=t.stride(),
      padding=t.padding(),
      function_prefix="h3_audio_up_\{i}",
    )
    units.push(
      unit("audio_up_\{i}", up.source, up.symbols, [
        ("input_time", t.input_time().to_int64()),
        ("output_time", t.output_time().to_int64()),
      ]),
    )
    let amp = @block.build_audio_amp(
      batch=t.decoder_batch_count(),
      channels=t.output_channels(),
      time=t.output_time(),
      function_prefix="h3_audio_amp_\{i}",
    )
    units.push(
      unit("audio_amp_\{i}", amp.source, amp.symbols, [
        ("channels", t.output_channels().to_int64()),
        ("time", t.output_time().to_int64()),
      ]),
    )
  }
  let last = transitions[transitions.length() - 1]
  let tail = @block.build_audio_tail(
    batch=last.decoder_batch_count(),
    channels=last.output_channels(),
    time=last.output_time(),
    function_prefix="h3_audio_tail",
  )
  units.push(
    unit("audio_tail", tail.source, tail.symbols, [
      ("time", tail.time.to_int64()),
    ]),
  )
}
