///|
fn read_video_config(
  root : @approved_fs.ApprovedRoot,
) -> @config.MiniMaxH3VideoVaeConfig raise {
  @config.parse_video_vae_checkpoint(
    @utf8.decode(read(root, "video_vae/config.json", 1048576L)),
    @utf8.decode(read(root, "video_vae/source/config.json", 1048576L)),
    max_bytes=1048576,
    max_depth=16,
  )
}

///|
fn video_statistics(table : ReadOnlyArray[Float]) -> Bytes {
  let bytes = @buffer.Buffer()
  for value in table {
    bytes.write_float_le(value)
  }
  bytes.to_bytes()
}

///|
/// Budget every allocation owned by the request, including its borrowed packed
/// input and two normalization tables. Module/queue reserve remains separate.
fn video_workspace_limit(
  device_limit : Int64,
  weight_bytes : Int64,
  input_bytes : Int64,
  statistics_bytes : Int64,
  module_reserve : Int64,
) -> Int64 raise {
  guard device_limit > 0L &&
    weight_bytes > 0L &&
    input_bytes > 0L &&
    statistics_bytes > 0L &&
    module_reserve > 0L else {
    fail("invalid VideoVAE budget")
  }
  for
    bytes in [weight_bytes, input_bytes, statistics_bytes, module_reserve]
    remaining = device_limit {
    guard bytes < remaining else { fail("VideoVAE device budget exhausted") }
    continue remaining - bytes
  } nobreak {
    remaining
  }
}

///|
/// Execute the existing full-request video decoder with real streamed weights.
/// Input is the packed, standardized F32 denoiser output; the shared bridge
/// performs inverse packing and checkpoint destandardization before decoding.
/// Standalone fixture execution is not full text-to-media generation.
async fn run_video(
  root_name : String,
  root : @approved_fs.ApprovedRoot,
  metadata : @h3.MiniMaxH3ModelMetadata,
  model : @joint.JointDiffusionPlan,
  args : ArrayView[String],
) -> Unit {
  let config = read_video_config(root)
  let shape = model
    .shapes()
    .resolve_request(
      height=@string.parse_int(args[1]),
      width=@string.parse_int(args[2]),
      requested_frames=@string.parse_int(args[3]),
      inference_steps=@string.parse_int(args[4]),
    )
  let plan = @loop.PreparedPlan::new(
    model,
    shape,
    max_latent_elements=2147483647L,
  )
  let _ = @vae.bind_video_latent_destandardization(metadata, config, shape)
  let means = video_statistics(config.latent_means())
  let stds = video_statistics(config.latent_standard_deviations())
  let input_size = plan.video_elements() * 4L
  let limit = @string.parse_int64(args[11])
  let component_root = @approved_fs.ApprovedRoot::open_absolute(
    root_name + "/video_vae/source",
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
    VideoVae,
    inventory,
    maximum_component_bytes=12884901888L,
    maximum_device_bytes=limit,
  )
  let weights_bytes = checkpoint.required_device_bytes()
  let reserve = 67108864L
  let workspace_limit = video_workspace_limit(
    limit,
    weights_bytes,
    input_size,
    (means.length() + stds.length()).to_int64(),
    reserve,
  )
  let bundle = @video.VideoDecodeBundle::from_plan(
    plan,
    prefix="h3_video",
    tile_size=@string.parse_int(args[5]),
    minimum_overlap=@string.parse_int(args[6]),
    max_tiles=@string.parse_int(args[7]),
    max_workspace_bytes=workspace_limit,
  )
  println(
    "video_weights=\{weights_bytes} video_workspace=\{bundle.workspace_bytes} video_input=\{input_size} pixels_bytes=\{bundle.output.output_bytes} device_limit=\{limit}",
  )
  let input_bytes = read(root, args[9], input_size)
  guard input_bytes.length().to_int64() == input_size else {
    fail("wrong packed video latent size")
  }
  let aot = @approved_fs.ApprovedRoot::open_absolute(args[8])
  defer aot.close()
  let binary = read(aot, "video_decode.cubin", reserve)
  let context = @device.open_context(0)
  defer context.close()
  let stream = context.create_stream()
  defer stream.close()
  let owner = @startup.WeightStartup::new(max_device_bytes=weights_bytes)
  defer owner.close()
  checkpoint.prepare(owner, context, component_root, chunk_bytes=1048576)
  let lease = owner.acquire()
  defer lease.release()
  let artifact = context.load_module(binary)
  defer artifact.close()
  let input = context.allocate(input_size)
  defer input.close()
  input.copy_from_host(input_bytes, 0L, 0L, input_size)
  let mean_allocation = context.allocate(means.length().to_int64())
  defer mean_allocation.close()
  mean_allocation.copy_from_host(means, 0L, 0L, means.length().to_int64())
  let std_allocation = context.allocate(stds.length().to_int64())
  defer std_allocation.close()
  std_allocation.copy_from_host(stds, 0L, 0L, stds.length().to_int64())
  let blocks = Array::makei(36, i => lease.with_allocation(1 + i, a => a))
  let request = @video.VideoDecodeRequest::new()
  defer request.close()
  request.prepare(
    context,
    stream,
    bundle,
    artifact,
    input~,
    input_offset=0L,
    frontend_weights=lease.with_allocation(0, a => a),
    blocks,
    head_weights=lease.with_allocation(37, a => a),
    means=mean_allocation,
    standard_deviations=std_allocation,
  )
  request.submit()
  request.wait()
  let pixels = request.with_output(output => {
    output.copy_to_host(0L, bundle.output.output_bytes)
  })
  @fs.write_file(args[10], pixels, create_mode=CreateNew, sync=Full)
  let (nonfinite, maximum) = waveform_statistics(pixels)
  println(
    "video_decoder_complete frames=\{bundle.output.frames} height=\{bundle.output.height} width=\{bundle.output.width} output_bytes=\{pixels.length()} nonfinite=\{nonfinite} maximum_absolute=\{maximum}",
  )
  guard nonfinite == 0 else {
    fail("nonfinite decoded pixels; output preserved")
  }
}
