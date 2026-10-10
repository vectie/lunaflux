///|
/// Executes real encoded text -> context/refiners -> all denoiser blocks ->
/// paired flow updates -> both actual VAEs. Encoder output is supplied explicitly
/// so its 51 GB weights need not coexist with the denoiser on unified memory.
/// This diagnostic is not an independent-reference parity or prompt-quality claim.
async fn run_joint(
  root_name : String,
  root : @approved_fs.ApprovedRoot,
  metadata : @h3.MiniMaxH3ModelMetadata,
  model : @joint.JointDiffusionPlan,
  args : ArrayView[String],
) -> Unit {
  let layout = joint_layout(root, metadata, model, args)
  let limit = @string.parse_int64(args[15])
  let transformer_root = @approved_fs.ApprovedRoot::open_absolute(
    root_name + "/transformer",
  )
  defer transformer_root.close()
  let audio_root = @approved_fs.ApprovedRoot::open_absolute(
    root_name + "/audio_vae",
  )
  defer audio_root.close()
  let video_root = @approved_fs.ApprovedRoot::open_absolute(
    root_name + "/video_vae/source",
  )
  defer video_root.close()
  let (denoiser_kind, _) = component("transformer", metadata.spec().variant())
  let denoiser = inspect_joint_component(
    root,
    transformer_root,
    metadata,
    model,
    denoiser_kind,
    args[8],
    limit,
  )
  let audio = inspect_joint_component(
    root,
    audio_root,
    metadata,
    model,
    AudioVae,
    args[9],
    limit,
  )
  let video = inspect_joint_component(
    root,
    video_root,
    metadata,
    model,
    VideoVae,
    args[10],
    limit,
  )
  // Reserve driver/modules, streaming transfer staging and host-side runtime
  // headroom explicitly. An external no-swap cgroup remains the physical cap.
  let reserve = 4294967296L
  let total = joint_sum_bytes(
    [
      denoiser.required_device_bytes(),
      audio.required_device_bytes(),
      video.required_device_bytes(),
      layout.workspace_bytes,
      reserve,
    ],
    limit,
  )
  let encoded_bytes = read(root, args[5], layout.encoded_bytes)
  guard encoded_bytes.length().to_int64() == layout.encoded_bytes else {
    fail("encoded text must be exact BF16 hidden_states[50] rows")
  }
  println(
    "joint_preflight weights=\{denoiser.required_device_bytes() + audio.required_device_bytes() + video.required_device_bytes()} workspace=\{layout.workspace_bytes} reserve=\{reserve} planned_total=\{total} device_limit=\{limit}",
  )
  let aot = @approved_fs.ApprovedRoot::open_absolute(args[7])
  defer aot.close()
  let context = @device.open_context(0)
  defer context.close()
  let stream = context.create_stream()
  defer stream.close()
  let denoiser_owner = @startup.WeightStartup::new(
    max_device_bytes=denoiser.required_device_bytes(),
  )
  defer denoiser_owner.close()
  let audio_owner = @startup.WeightStartup::new(
    max_device_bytes=audio.required_device_bytes(),
  )
  defer audio_owner.close()
  let video_owner = @startup.WeightStartup::new(
    max_device_bytes=video.required_device_bytes(),
  )
  defer video_owner.close()
  denoiser.prepare(
    denoiser_owner,
    context,
    transformer_root,
    chunk_bytes=1048576,
  )
  audio.prepare(audio_owner, context, audio_root, chunk_bytes=1048576)
  video.prepare(video_owner, context, video_root, chunk_bytes=1048576)
  let denoiser_lease = denoiser_owner.acquire()
  defer denoiser_lease.release()
  let audio_lease = audio_owner.acquire()
  defer audio_lease.release()
  let video_lease = video_owner.acquire()
  defer video_lease.release()
  let modules : JointModules = {
    context,
    root: aot,
    modules: [],
    functions: [],
    loaded: 0L,
    maximum: reserve / 2L,
  }
  defer modules.close()
  let storage = @latent.LatentStorage::allocate(context, layout.storage)
  defer storage.close()
  storage.initialize_noise(
    seed=@string.parse_int64(args[16]),
    chunk_elements=262144,
  )
  ignore(storage.upload_flow_table(context))
  pack_joint_noise(context, stream, storage, modules)
  let encoded = upload_joint_bytes(context, encoded_bytes)
  defer encoded.close()
  let rotary_allocation = context.allocate(layout.rotary_bytes)
  defer rotary_allocation.close()
  let rotary = @latent.RotaryBindings::upload(
    context,
    rotary_allocation,
    layout.rotary.cosine()[:],
    layout.rotary.sine()[:],
    rows=layout.sequence.assembly().rows().length(),
    rotary_width=layout.rotary.width(),
    offset=0L,
    chunk_elements=262144,
  )
  let embeddings = context.allocate(layout.embedding_bytes)
  defer embeddings.close()
  let parameters = context.allocate(layout.parameter_bytes)
  defer parameters.close()
  let text = @program.TextConditioningProgram::new(layout.refiner)
  defer text.close()
  let conditioning = @program.ConditioningProgram::new(layout.conditioning)
  defer conditioning.close()
  let input = @program.InputProgram::new(layout.input)
  defer input.close()
  input.prepare(
    context,
    modules.load("input"),
    storage,
    max_projection_bytes=layout.projection_bytes,
  )
  defer layout.metadata.close()
  layout.metadata.prepare(context)
  defer layout.predictions.close()
  let prediction_modules = layout.metadata.distinct_timestep_counts()[:].map(n => {
    (
      modules.load("dit_n\{n}"),
      modules.load("timestep_n\{n}"),
      modules.load("output_n\{n}"),
    )
  })
  layout.predictions.prepare(context, prediction_modules[:])
  let variants = layout.predictions.variants()
  let audio_config = read_audio_config(root)
  let video_config = read_video_config(root)
  let audio_means = upload_joint_bytes(
    context,
    video_statistics(audio_config.latent_means()),
  )
  defer audio_means.close()
  let audio_stds = upload_joint_bytes(
    context,
    video_statistics(audio_config.latent_standard_deviations()),
  )
  defer audio_stds.close()
  let video_means = upload_joint_bytes(
    context,
    video_statistics(video_config.latent_means()),
  )
  defer video_means.close()
  let video_stds = upload_joint_bytes(
    context,
    video_statistics(video_config.latent_standard_deviations()),
  )
  defer video_stds.close()
  println("joint_statistics_bytes=\{layout.statistics_bytes}")
  let audio_input_module = modules.load("audio_input")
  let permutation = audio_input_module.load_function(
    @utf8.encode("lunaflux_dense_permutation_u32_v1"),
  )
  defer permutation.close()
  let destandardize = audio_input_module.load_function(
    @utf8.encode("lunaflux_audio_destandardize_v1"),
  )
  defer destandardize.close()
  let audio_input = @program.AudioInputProgram::allocate(
    context,
    permutation,
    destandardize,
    time=layout.plan.request().audio_latent_frames(),
    standard_deviations=audio_stds,
    means=audio_means,
    max_bytes=layout.audio.memory().input_bytes * 2L,
  )
  defer audio_input.close()
  defer layout.audio.close()
  layout.audio.prepare(context, stream, audio_lease, name => {
    let bytes = read(aot, name + ".cubin", modules.maximum - modules.loaded)
    modules.loaded += bytes.length().to_int64()
    context.load_module(bytes)
  })
  let audio_request = @program.AudioDecodeRequest::new()
  defer audio_request.close()
  layout.audio.prepare_joint_request(
    audio_request,
    context,
    stream,
    layout.plan,
    storage,
    audio_input,
  )
  let video_request = @video.VideoDecodeRequest::new()
  defer video_request.close()
  let video_blocks = Array::makei(video_config.decoder_layer_count(), i => {
    video_lease.with_allocation(1 + i, a => a)
  })
  let video_module = modules.load("video_decode")
  let video_front = video_lease.with_allocation(0, a => a)
  let video_head = video_lease.with_allocation(1 + video_blocks.length(), a => a)
  storage.with_region(VideoState, (allocation, offset, _) => {
    video_request.prepare(
      context,
      stream,
      layout.video,
      video_module,
      input=allocation,
      input_offset=offset,
      frontend_weights=video_front,
      video_blocks,
      head_weights=video_head,
      means=video_means,
      standard_deviations=video_stds,
    )
  })
  let request = @request.Request::new()
  defer request.close()
  let refiner_module = modules.load("refiner")
  let conditioning_module = modules.load("conditioning")
  let video_update = modules.function(
    "flow_video", "lunaflux_rectified_flow_f32_eta0_v1",
  )
  let audio_update = modules.function(
    "flow_audio", "lunaflux_rectified_flow_f32_eta0_v1",
  )
  denoiser_lease.with_denoiser_weights(denoiser_kind, model, (
    weights,
    context_weights,
    refiners,
  ) => {
    text.prepare(
      context,
      stream,
      refiner_module,
      refiners[:],
      encoded~,
      encoded_offset=0L,
      weights=context_weights,
      max_workspace_bytes=layout.refiner_workspace,
      max_output_bytes=layout.text_bytes,
    )
    conditioning.prepare(
      context,
      stream,
      conditioning_module,
      text,
      [],
      max_output_bytes=layout.conditioning.byte_count,
    )
    request.prepare_model(
      context,
      stream,
      layout.plan,
      storage,
      [@execution.PreparedStage::own(text)],
      conditioning,
      layout.metadata,
      variants[:],
      input,
      rotary,
      weights,
      video_request,
      audio_request,
      timestep_embeddings=embeddings,
      timestep_embeddings_offset=0L,
      parameters~,
      parameter_offset=0L,
      video_update~,
      audio_update~,
      video_block_size=256,
      audio_block_size=256,
    )
  })
  for ;; {
    let stage = request.advance()
    println("joint_stage=\{Repr(stage)}")
    match stage {
      Completed => break
      Cancelled | Failed | Closed => fail("joint request did not complete")
      _ => ()
    }
  }
  guard request.outputs_ready() else { fail("joint outputs not ready") }
  let pixels = request.with_video(a => {
    a.copy_to_host(0L, layout.video.output.output_bytes)
  })
  let waveform = request.with_audio(a => {
    a.copy_to_host(0L, layout.audio.memory().output_bytes)
  })
  @fs.mkdir(args[17])
  @fs.write_file(
    args[17] + "/video.f32",
    pixels,
    create_mode=CreateNew,
    sync=Full,
  )
  @fs.write_file(
    args[17] + "/audio.f32",
    waveform,
    create_mode=CreateNew,
    sync=Full,
  )
  let (bad_pixels, _) = waveform_statistics(pixels)
  let (bad_samples, _) = waveform_statistics(waveform)
  println(
    "joint_complete steps=\{layout.plan.steps().length()} denoiser_layers=\{model.denoiser().layer_count()} video_bytes=\{pixels.length()} audio_bytes=\{waveform.length()} audio_samples=\{request.waveform_samples()} nonfinite=\{bad_pixels + bad_samples}",
  )
  guard bad_pixels == 0 && bad_samples == 0 else {
    fail("nonfinite joint output; diagnostics preserved")
  }
}
