///|
/// Generate all AOT modules for a resolved request. Scheduling metadata decides
/// every required timestep variant; no hand-maintained variant count is used.
/// Optional vision geometry describes Qwen-VL inputs, not video VAE inputs.
pub fn build(
  metadata : @spec.MiniMaxH3ModelMetadata,
  plan : @loop.PreparedPlan,
  sequence : @sequence.Sequence,
  audio_config : @config.MiniMaxH3AudioVaeConfig,
  text_plan : @text.TextEncoderPlan,
  text_config~ : @text.EncoderConfig,
  visual_noise~ : Double,
  audio_noise~ : Double,
  vision_text_rows~ : ArrayView[Int],
  max_metadata_bytes~ : Int64,
  max_workspace_bytes~ : Int64,
  tile_size~ : Int,
  minimum_overlap~ : Int,
  max_tiles~ : Int,
  vision? : @vision.VisionPlan,
) -> SourceExport raise {
  let model = @h3.build(metadata)
  guard text_plan.epsilon == text_config.epsilon &&
    text_plan.rows <= text_config.max_positions else {
    raise InvalidPlan
  }
  let vision_rows : Map[Int, Unit] = Map([])
  for row in vision_text_rows {
    guard row >= 0 && row < text_plan.rows && !vision_rows.contains(row) else {
      raise InvalidPlan
    }
    vision_rows[row] = ()
  }
  match vision {
    Some(value) => {
      guard value.merged_rows == vision_text_rows.length() else {
        raise InvalidPlan
      }
    }
    None => {
      guard vision_text_rows.is_empty() else { raise InvalidPlan }
    }
  }
  guard model.plan_digest() == plan.model_digest() &&
    sequence.segments().length() > 0 &&
    sequence.assembly().width() == model.denoiser().hidden_size() &&
    sequence.segments()[0].kind is Text &&
    sequence.segments()[0].count == text_plan.rows else {
    raise InvalidPlan
  }
  let request = plan.request()
  let shape = model.shapes()
  let expected_video_rows = request.video_latent_frames().to_int64() /
    shape.video_patch_temporal().to_int64() *
    (
      request.video_latent_height().to_int64() /
      shape.video_patch_height().to_int64()
    ) *
    (
      request.video_latent_width().to_int64() /
      shape.video_patch_width().to_int64()
    )
  let mut video_rows = 0L
  let mut audio_rows = 0L
  for segment in sequence.segments() {
    match segment.kind {
      TargetVideo => video_rows += segment.count.to_int64()
      TargetAudio => audio_rows += segment.count.to_int64()
      _ => ()
    }
  }
  guard video_rows == expected_video_rows &&
    audio_rows == request.audio_token_rows().to_int64() else {
    raise InvalidPlan
  }
  let steps = plan.steps()
  let schedule = sequence.schedule(
    video=steps[:].map(s => s.video_timestep),
    audio=steps[:].map(s => s.audio_timestep),
    visual_noise~,
    audio_noise~,
    vision_text_rows~,
    max_steps=steps.length(),
    max_index_bytes=max_metadata_bytes,
  )
  let counts : Array[Int] = []
  for step in schedule.steps() {
    if !counts.contains(step.values.length()) {
      counts.push(step.values.length())
    }
  }
  counts.sort()
  let units : Array[SourceUnit] = []
  for
    (name, elements) in [
      ("flow_video", plan.video_elements()),
      ("flow_audio", plan.audio_elements()),
    ] {
    units.push(
      unit(
        name,
        @flow.render(elements~, block_size=256),
        ["lunaflux_rectified_flow_f32_eta0_v1"],
        [("elements", elements), ("block_size", 256L)],
      ),
    )
  }
  units.push(
    unit(
      "video_pack",
      @permutation.render(plan.video_pack(), block_size=256),
      ["lunaflux_dense_permutation_u32_v1"],
      [("elements", plan.video_elements()), ("block_size", 256L)],
    ),
  )
  let rows = sequence.assembly().rows().length().to_int64()
  let epsilon = metadata.spec().norm_epsilon()
  for count in counts {
    let name = "dit_n\{count}"
    let source = @block.build(
      model,
      sequence,
      distinct_timesteps=count,
      block_norm_epsilon=epsilon,
      function_prefix="h3_block_n\{count}",
    )
    units.push(
      unit(name, source.source, source.symbols, [
        ("rows", rows),
        ("timesteps", count.to_int64()),
      ]),
    )
    let time = @block.build_timestep(
      model,
      distinct_timesteps=count,
      function_prefix="h3_timestep_n\{count}",
    )
    units.push(
      unit("timestep_n\{count}", time.source, time.symbols, [
        ("rows", count.to_int64()),
      ]),
    )
    let output = @block.build_output(
      model,
      sequence,
      distinct_timesteps=count,
      function_prefix="h3_output_n\{count}",
    )
    units.push(
      unit("output_n\{count}", output.source, output.symbols, [
        ("rows", rows),
        ("timesteps", count.to_int64()),
      ]),
    )
  }
  let input = @block.build_input(model, sequence, function_prefix="h3_input")
  units.push(unit("input", input.source, input.symbols, [("rows", rows)]))
  let conditioning = @block.build_conditioning(
    model,
    sequence,
    function_prefix="h3_conditioning",
  )
  units.push(
    unit("conditioning", conditioning.source, conditioning.symbols, [
      ("bytes", conditioning.byte_count),
    ]),
  )
  let refiner = @block.build_refiner(
    model,
    text_rows=text_plan.rows,
    block_norm_epsilon=epsilon,
    function_prefix="h3_refiner",
  )
  units.push(
    unit("refiner", refiner.source, refiner.symbols, [
      ("rows", text_plan.rows.to_int64()),
    ]),
  )
  let text = @text_source.build_source(text_plan, prefix="h3_text")
  units.push(
    unit("text", text.source, text.symbols, [
      ("rows", text_plan.rows.to_int64()),
      ("hidden_state", 50L),
    ]),
  )
  if vision is Some(vision) {
    let v = @vision_source.build_source(vision, prefix="h3_vision")
    units.push(
      unit("vision", v.source, v.symbols, [
        ("rows", vision.rows.to_int64()),
        ("merged_rows", vision.merged_rows.to_int64()),
      ]),
    )
  }
  append_audio(units, metadata, audio_config, plan)
  let decoder = @video.TiledDecoder::new(
    latent_frames=request.video_latent_frames(),
    latent_height=request.video_latent_height(),
    latent_width=request.video_latent_width(),
    output_frames=request.frames(),
    tile_size~,
    minimum_overlap~,
    prefix="h3_video",
    max_jobs=max_tiles,
    max_workspace_bytes~,
  )
  let video_input = @block.build_video_input(plan)
  let bundle = @video.VideoDecodeBundle::new(
    video_input,
    decoder,
    max_workspace_bytes~,
  )
  let symbols = video_input.symbols[:].to_owned()
  symbols.append(decoder.tile.frontend.symbols[:])
  symbols.append(decoder.tile.block.symbols[:])
  symbols.append(decoder.tile.head.symbols[:])
  symbols.append(decoder.symbols[:])
  symbols.push(bundle.output.symbol)
  units.push(
    unit("video_decode", bundle.source, ReadOnlyArray::from_array(symbols), [
      ("frames", request.frames().to_int64()),
      ("height", request.height().to_int64()),
      ("width", request.width().to_int64()),
      ("workspace_bytes", bundle.workspace_bytes),
    ]),
  )
  {
    model_digest: model.plan_digest().as_hex(),
    units: ReadOnlyArray::from_array(units),
    timestep_counts: ReadOnlyArray::from_array(counts),
  }
}
