///|
fn read_audio_config(
  root : @approved_fs.ApprovedRoot,
) -> @config.MiniMaxH3AudioVaeConfig raise {
  @config.parse_audio_vae_checkpoint(
    @utf8.decode(read(root, "audio_vae/config.json", 1048576L)),
    @utf8.decode(read(root, "audio_vae/metadata.json", 1048576L)),
    @utf8.decode(read(root, "audio_vae/config.yaml", 1048576L)),
    max_bytes=1048576,
    max_depth=16,
  )
}

///|
/// Standalone actual-checkpoint AudioVAE execution. Input is a caller-supplied
/// channel-major destandardized F32 latent plane, not fabricated conditioning.
/// Full joint denoising and video output remain a separate prepared request.
async fn run_audio(
  root_name : String,
  root : @approved_fs.ApprovedRoot,
  metadata : @h3.MiniMaxH3ModelMetadata,
  model : @joint.JointDiffusionPlan,
  args : ArrayView[String],
) -> Unit {
  let config = read_audio_config(root)
  let shape = model
    .shapes()
    .resolve_request(
      height=@string.parse_int(args[1]),
      width=@string.parse_int(args[2]),
      requested_frames=@string.parse_int(args[3]),
      inference_steps=@string.parse_int(args[4]),
    )
  let transitions = @vae.bind_audio_decoder_upsample_stages(
    metadata, config, shape,
  )
  let sources = @source.build_audio_decoder(transitions[:])
  let device_limit = @string.parse_int64(args[8])
  let component_root = @approved_fs.ApprovedRoot::open_absolute(
    root_name + "/audio_vae",
  )
  defer component_root.close()
  let inventory = @inventory.CheckpointInventory::parse(
    @utf8.decode(read(root, args[0], 1048576L)),
    maximum_bytes=1048576,
    maximum_shards=64,
  )
  let checkpoint = @checkpoint.ComponentCheckpoint::inspect(
    component_root,
    metadata,
    model,
    AudioVae,
    inventory,
    maximum_component_bytes=1073741824L,
    maximum_device_bytes=device_limit,
  )
  let weights_bytes = checkpoint.required_device_bytes()
  let module_reserve = 67108864L
  guard device_limit > weights_bytes &&
    device_limit - weights_bytes > module_reserve else {
    fail("AudioVAE device budget excludes decoder workspace")
  }
  let decoder = @audio.CheckpointAudioDecoder::new(
    sources,
    maximum_workspace_bytes=device_limit - weights_bytes - module_reserve,
  )
  let memory = decoder.memory()
  println(
    "audio_weights=\{weights_bytes} audio_workspace=\{memory.total_bytes} audio_input=\{memory.input_bytes} waveform_bytes=\{memory.output_bytes} device_limit=\{device_limit}",
  )
  let input_bytes = read(root, args[6], memory.input_bytes)
  guard input_bytes.length().to_int64() == memory.input_bytes else {
    fail("wrong latent input size")
  }
  let aot = @approved_fs.ApprovedRoot::open_absolute(args[5])
  defer aot.close()
  let context = @device.open_context(0)
  defer context.close()
  let stream = context.create_stream()
  defer stream.close()
  let owner = @startup.WeightStartup::new(max_device_bytes=weights_bytes)
  defer owner.close()
  checkpoint.prepare(owner, context, component_root, chunk_bytes=1048576)
  let lease = owner.acquire()
  defer lease.release()
  // Decoder must retire before its borrowed checkpoint lease and GPU context.
  defer decoder.close()
  let mut module_bytes = 0L
  decoder.prepare(context, stream, lease, name => {
    let bytes = read(aot, name + ".cubin", module_reserve - module_bytes)
    module_bytes += bytes.length().to_int64()
    context.load_module(bytes)
  })
  let input = context.allocate(memory.input_bytes)
  defer input.close()
  input.copy_from_host(input_bytes, 0L, 0L, memory.input_bytes)
  let request = @program.AudioDecodeRequest::new()
  defer request.close()
  decoder.prepare_request(request, context, stream, input)
  request.execute()
  let waveform = request.with_output(output => {
    output.copy_to_host(0L, memory.output_bytes)
  })
  @fs.write_file(args[7], waveform, create_mode=CreateNew, sync=Full)
  let (nonfinite, maximum) = waveform_statistics(waveform)
  println(
    "audio_decoder_complete samples_per_channel=\{request.waveform_samples()} output_bytes=\{waveform.length()} nonfinite=\{nonfinite} maximum_absolute=\{maximum}",
  )
  guard nonfinite == 0 else {
    fail("nonfinite decoded waveform; output preserved")
  }
}
