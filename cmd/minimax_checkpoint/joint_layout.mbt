///|
// All geometry and residency checks run before a GPU context is opened. This
// is a text-only diagnostic request: reference-media rows are not fabricated.
struct JointLayout {
  plan : @loop.PreparedPlan
  sequence : @sequence.Sequence
  storage : @loop.LatentStoragePlan
  metadata : @program.PredictionMetadata
  predictions : @program.PredictionPrograms
  refiner : @source.RefinerSource
  conditioning : @source.ConditioningSource
  input : @source.InputSource
  video : @video.VideoDecodeBundle
  audio : @audio.CheckpointAudioDecoder
  rotary : @sequence.RotaryTables
  projection_bytes : Int64
  refiner_workspace : Int64
  encoded_bytes : Int64
  text_bytes : Int64
  embedding_bytes : Int64
  parameter_bytes : Int64
  rotary_bytes : Int64
  statistics_bytes : Int64
  workspace_bytes : Int64
}

///|
fn joint_sum_bytes(parts : ArrayView[Int64], maximum : Int64) -> Int64 raise {
  guard maximum > 0L else { fail("invalid aggregate joint budget") }
  for bytes in parts; total = 0L {
    guard bytes > 0L && bytes <= maximum - total else {
      fail("aggregate joint device budget exhausted before allocation")
    }
    continue total + bytes
  } nobreak {
    total
  }
}

///|
fn joint_f32_values(bytes : Bytes) -> Array[Float] raise {
  guard bytes.length() > 0 && bytes.length() % 4 == 0 else {
    fail("truncated F32 coefficients")
  }
  Array::makei(bytes.length() / 4, i => {
    let offset = i * 4
    let bits = bytes[offset].to_uint() |
      (bytes[offset + 1].to_uint() << 8) |
      (bytes[offset + 2].to_uint() << 16) |
      (bytes[offset + 3].to_uint() << 24)
    let value = Float::reinterpret_from_uint(bits)
    guard !value.is_nan() && !value.is_inf() else {
      fail("nonfinite checkpoint coefficient")
    }
    value
  })
}

///|
fn joint_layout(
  root : @approved_fs.ApprovedRoot,
  metadata : @h3.MiniMaxH3ModelMetadata,
  model : @joint.JointDiffusionPlan,
  args : ArrayView[String],
) -> JointLayout raise {
  let maximum = @string.parse_int64(args[14])
  let rows = @string.parse_int(args[4])
  let shape = model
    .shapes()
    .resolve_request(
      height=@string.parse_int(args[0]),
      width=@string.parse_int(args[1]),
      requested_frames=@string.parse_int(args[2]),
      inference_steps=@string.parse_int(args[3]),
    )
  let plan = @loop.PreparedPlan::new(
    model,
    shape,
    max_latent_elements=2147483647L,
  )
  let metadata_limit = Int64::min(maximum, 1073741824L)
  let max_rows = request_source_row_limit(
    maximum,
    metadata_limit,
    model.denoiser().hidden_size(),
  )
  let grid = @sequence.VideoGrid::new(
    frames=shape.video_latent_frames(),
    height=shape.video_latent_height(),
    width=shape.video_latent_width(),
    max_rows~,
  )
  let positions = match metadata.spec().variant() {
    Fl2va =>
      @sequence.PositionedSequence::keyframes(
        grid,
        hidden_width=model.denoiser().hidden_size(),
        text_rows=rows,
        audio_frames=shape.audio_latent_frames(),
        anchors=[],
        max_rows~,
      )
    Ref2va =>
      @sequence.PositionedSequence::references(
        grid,
        hidden_width=model.denoiser().hidden_size(),
        text_rows=rows,
        audio_frames=shape.audio_latent_frames(),
        references=[],
        max_rows~,
      )
  }
  let sequence = positions.sequence()
  let rotary = positions.rotary_tables(
    joint_f32_values(read(root, args[6], 1048576L))[:],
    max_elements=maximum / 8L,
  )
  let storage = @loop.LatentStoragePlan::new(plan, max_bytes=maximum)
  let projections = storage.projections(model, max_bytes=maximum)
  let time = @program.PredictionMetadata::new(
    plan,
    sequence,
    visual_noise=0.0,
    audio_noise=0.0,
    vision_text_rows=[],
    max_metadata_bytes=metadata_limit,
  )
  let counts = time.distinct_timestep_counts()
  let sources = counts[:].map(n => {
    @source.build_prediction_sources(
      model,
      sequence,
      distinct_timesteps=n,
      block_norm_epsilon=metadata.spec().norm_epsilon(),
      function_prefix="h3",
    )
  })
  let predictions = @program.PredictionPrograms::new(
    sources[:],
    max_workspace_bytes=maximum,
  )
  let refiner = @source.build_refiner(
    model,
    text_rows=rows,
    block_norm_epsilon=metadata.spec().norm_epsilon(),
    function_prefix="h3_refiner",
  )
  let refiner_workspace = @latent.BlockWorkspaceLayout::new(
      rows~,
      hidden_width=model.denoiser().hidden_size(),
      head_count=model.denoiser().attention_head_count(),
      head_dimension=model.denoiser().attention_head_dimension(),
      intermediate_width=model.denoiser().feed_forward_dimension(),
      max_bytes=maximum,
    ).byte_count
  let audio_config = read_audio_config(root)
  let audio = @audio.CheckpointAudioDecoder::new(
    @source.build_audio_decoder(
      @vae.bind_audio_decoder_upsample_stages(metadata, audio_config, shape)[:],
    ),
    maximum_workspace_bytes=maximum,
  )
  let video = @video.VideoDecodeBundle::from_plan(
    plan,
    prefix="h3_video",
    tile_size=@string.parse_int(args[11]),
    minimum_overlap=@string.parse_int(args[12]),
    max_tiles=@string.parse_int(args[13]),
    max_workspace_bytes=maximum,
  )
  let conditioning = @source.build_conditioning(
    model,
    sequence,
    function_prefix="h3_conditioning",
  )
  let text_bytes = rows.to_int64() *
    model.denoiser().hidden_size().to_int64() *
    2L
  let encoded_bytes = rows.to_int64() *
    model.denoiser().text_conditioning_dimension().to_int64() *
    2L
  let max_count = counts[counts.length() - 1]
  let embedding_bytes = max_count.to_int64() *
    model.denoiser().final_adaln_conditioning_dimension().to_int64() *
    2L
  let parameter_bytes = @latent.ParameterProjectionLayout::new(
      rows=max_count,
      input_width=model.denoiser().final_adaln_conditioning_dimension(),
      width=model.denoiser().hidden_size(),
      groups=3,
      fields=6,
    ).output_bytes
  let packed_bytes = sequence.assembly().element_count() * 2L
  let packed_total = packed_bytes +
    (256L - packed_bytes % 256L) % 256L +
    sequence.assembly().mapping().length().to_int64()
  let rotary_bytes = rotary.cosine().length().to_int64() * 8L
  let video_config = read_video_config(root)
  let statistics_bytes = (video_config.latent_means().length() +
    video_config.latent_standard_deviations().length() +
    audio_config.latent_means().length() +
    audio_config.latent_standard_deviations().length()).to_int64() *
    4L
  let workspace_bytes = joint_sum_bytes(
    [
      storage.byte_count(),
      projections.byte_count(),
      packed_total,
      time.byte_count(),
      predictions.workspace_bytes(),
      refiner_workspace,
      text_bytes,
      encoded_bytes,
      conditioning.byte_count,
      embedding_bytes,
      parameter_bytes,
      rotary_bytes,
      video.workspace_bytes,
      audio.joint_workspace_bytes(),
      statistics_bytes,
    ],
    @string.parse_int64(args[15]),
  )
  {
    plan,
    sequence,
    storage,
    metadata: time,
    predictions,
    refiner,
    conditioning,
    input: @source.build_input(model, sequence, function_prefix="h3_input"),
    video,
    audio,
    rotary,
    projection_bytes: projections.byte_count(),
    refiner_workspace,
    encoded_bytes,
    text_bytes,
    embedding_bytes,
    parameter_bytes,
    rotary_bytes,
    statistics_bytes,
    workspace_bytes,
  }
}
