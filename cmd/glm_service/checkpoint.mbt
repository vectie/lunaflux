///|
fn checkpoint_limits(
  cap : Int64,
) -> @streaming.StreamingSafetensorsLimits raise {
  @streaming.StreamingSafetensorsLimits::new(
    maximum_shards=512,
    maximum_file_bytes=17179869184L,
    maximum_header_bytes=33554432,
    maximum_tensors_per_shard=500000,
    maximum_total_tensors=500000,
    maximum_rank=8,
    maximum_name_bytes=512,
    maximum_dtype_specs=9,
    maximum_copy_requests=20000,
    maximum_copy_bytes=cap,
  )
}

///|
fn inspect_checkpoint(
  root : @approved_fs.ApprovedRoot,
  kernels : @approved_fs.ApprovedRoot,
  plan : ServicePlan,
) -> @materialize.PackedCheckpoint raise {
  let inventory = @inventory.CheckpointInventory::parse(
    @utf8.decode(read_bounded(kernels, plan.inventory, 1048576L)),
    maximum_bytes=1048576,
    maximum_shards=512,
  )
  let dtypes = [
    @streaming.StreamingSafetensorsDTypeSpec::bfloat16(),
    @streaming.StreamingSafetensorsDTypeSpec::float32(),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_fixed_width(
      "F16",
      bytes_per_element=2L,
    ),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("U8"),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("I8"),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_fixed_width(
      "I32",
      bytes_per_element=4L,
    ),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_fixed_width(
      "I64",
      bytes_per_element=8L,
    ),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3"),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3FN"),
  ]
  @materialize.PackedCheckpoint::new(
    @streaming.inspect_shards(
      root,
      inventory.shards(),
      dtype_specs=dtypes,
      limits=checkpoint_limits(plan.budget),
    ),
  )
}

///|
fn inspect_spec(root : @approved_fs.ApprovedRoot) -> @spec.Glm53FlashSpec raise {
  @config.parse_flash_spec_bytes(
    read_bounded(root, "config.json", 1048576L),
    max_bytes=1048576,
    max_depth=16,
  )
}

///|
fn prepare_rank(
  plan : ServicePlan,
  spec : @spec.Glm53FlashSpec,
  ordinal : Int,
  context : @device.Context,
  stream : @device.Stream,
  aot : @device.Module,
  sampling : @device.Module,
  root : @approved_fs.ApprovedRoot,
  checkpoint : @materialize.PackedCheckpoint,
  stage : @glm.GlmDecoderStage,
  rank : @rank.BatchedDecoderRank,
  collective : @device.TensorCollectiveOwner,
) -> Unit raise {
  let bytes = stage.required_device_bytes(checkpoint)
  guard bytes <= plan.budget - plan.reserve &&
    rank.required_device_bytes() <= plan.reserve / 8L else {
    fail("GLM owner exceeds per-Spark envelope")
  }
  // Rank zero's stdout is the framed worker channel, never a log stream.
  if ordinal == 1 {
    println(
      "GLM rank=\{ordinal} planned_total=\{bytes + plan.reserve} budget=\{plan.budget} rows=\{plan.rows} slots=\{plan.slots} context=\{plan.context} physical_pages=\{plan.history().geometry().1} vocabulary=\{spec.vocabulary_size()} \{plan.sparse_description()} payload_hashing=false",
    )
  }
  rank.prepare_ports(context, maximum_device_bytes=plan.reserve / 8L)
  let limits = checkpoint_limits(plan.budget)
  checkpoint.with_transfer_session(root, limits, bank => {
    stage.prepare_inputs(
      context,
      aot,
      stream,
      bank,
      root,
      @glm.GlmDecoderInputs::from_batched_rank(rank),
      maximum_device_bytes=plan.budget - plan.reserve,
      chunk_bytes=1048576,
      limits~,
      collective~,
      retirement=WholeFrame,
      replay=CaptureDecodeCollectives,
      replay_budget_bytes=plan.reserve / 4L,
      overlap_experts=true,
      fuse_hyper=true,
      fuse_controls=true,
      defer_hyper=true,
    )
  })
  stage.bind_batched_rank(
    rank,
    context,
    sampling_aot=sampling,
    maximum_sampling_device_bytes=plan.reserve / 8L,
    maximum_sampling_host_bytes=plan.reserve / 8L,
  )
}
