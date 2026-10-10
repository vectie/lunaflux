///|
/// Offline geometry only: no checkpoint payload, CUDA context, or JIT is opened.
/// Original metadata feeds the same full-request source builder as execution.
async fn export_request(
  root : @approved_fs.ApprovedRoot,
  metadata : @h3.MiniMaxH3ModelMetadata,
  model : @joint.JointDiffusionPlan,
  args : ArrayView[String],
) -> Unit {
  let options = request_export_arguments(args)
  let audio = read_audio_config(root)
  ignore(read_video_config(root))
  let text = @text.parse_config(
    read(root, "text_encoder/config.json", 1048576L),
    max_bytes=1048576,
  )
  guard options.text_rows <= text.max_positions else {
    fail("text rows exceed encoder position capacity")
  }
  let request = model
    .shapes()
    .resolve_request(
      height=options.height,
      width=options.width,
      requested_frames=options.frames,
      inference_steps=options.steps,
    )
  let plan = @loop.PreparedPlan::new(
    model,
    request,
    max_latent_elements=2147483647L,
  )
  let text_plan = @text.TextEncoderPlan::new(
    rows=options.text_rows,
    epsilon=text.epsilon,
  )
  let max_metadata_bytes = 1073741824L
  // Each row needs BF16 hidden storage and two I32 indexing fields. Derive the
  // bound from the explicit budgets, rather than passing Int::max_value to a
  // bounded sequence planner (which rejects it even for tiny requests).
  let max_rows = request_source_row_limit(
    options.workspace_bytes,
    max_metadata_bytes,
    model.denoiser().hidden_size(),
  )
  let sequence = @sequence.Sequence::from_request(
    model,
    request,
    text_rows=options.text_rows,
    conditioning=[],
    max_rows~,
  )
  let sources = @export.build(
    metadata,
    plan,
    sequence,
    audio,
    text_plan,
    text_config=text,
    visual_noise=0.0,
    audio_noise=0.0,
    vision_text_rows=[],
    max_metadata_bytes~,
    max_workspace_bytes=options.workspace_bytes,
    tile_size=options.tile_size,
    minimum_overlap=options.overlap,
    max_tiles=options.max_tiles,
  )
  sources.write(options.output_directory)
  println(
    "request_source_modules=\{sources.units.length()} text_rows=\{options.text_rows} frames=\{request.frames()} steps=\{options.steps}",
  )
  for unit in sources.units {
    println("module=\{unit.name} source_bytes=\{unit.source.length()}")
  }
}

///|
fn request_source_row_limit(
  workspace_bytes : Int64,
  metadata_bytes : Int64,
  hidden_width : Int,
) -> Int raise {
  guard workspace_bytes > 0L &&
    metadata_bytes > 0L &&
    metadata_bytes <= 2147483647L &&
    hidden_width > 0 else {
    fail("invalid request source storage capacity")
  }
  let rows = (workspace_bytes / (hidden_width.to_int64() * 2L)).min(
    metadata_bytes / (2L * 4L),
  )
  guard rows > 0L else { fail("request source budget cannot hold one row") }
  rows.to_int()
}

///|
priv struct RequestExportArguments {
  height : Int
  width : Int
  frames : Int
  steps : Int
  text_rows : Int
  tile_size : Int
  overlap : Int
  max_tiles : Int
  workspace_bytes : Int64
  output_directory : String
}

///|
fn request_export_arguments(
  args : ArrayView[String],
) -> RequestExportArguments raise {
  guard args
    is [
      height,
      width,
      frames,
      steps,
      rows,
      tile,
      overlap,
      tiles,
      workspace,
      output,
    ] else {
    fail("wrong request export argument count")
  }
  let options = {
    height: @string.parse_int(height),
    width: @string.parse_int(width),
    frames: @string.parse_int(frames),
    steps: @string.parse_int(steps),
    text_rows: @string.parse_int(rows),
    tile_size: @string.parse_int(tile),
    overlap: @string.parse_int(overlap),
    max_tiles: @string.parse_int(tiles),
    workspace_bytes: @string.parse_int64(workspace),
    output_directory: output,
  }
  guard options.height > 0 &&
    options.width > 0 &&
    options.frames > 0 &&
    options.steps > 0 &&
    options.text_rows > 0 &&
    options.tile_size > 0 &&
    options.overlap >= 0 &&
    options.overlap < options.tile_size &&
    options.max_tiles > 0 &&
    options.workspace_bytes > 0L &&
    !output.is_empty() else {
    fail("invalid request export geometry or budget")
  }
  options
}
