///|
/// Startup-only checkpoint bridge. The complete semantic manifest feeds the
/// existing bounded uploader; no whole-component host arena is materialized.
pub struct ComponentCheckpoint {
  priv manifest : @host.MiniMaxH3HostComponentManifest
  priv sources : Array[@host.MiniMaxH3ShardSource]
  priv packings : Array[@startup.Packing]
  priv limits : @host.MiniMaxH3HostMaterializationLimits
  priv bytes : Int64
}

///|
pub fn component_packings(
  component : @weights.MiniMaxH3WeightComponent,
  plan : @joint.JointDiffusionPlan,
  maximum_device_bytes~ : Int64,
) -> Array[@startup.Packing] raise {
  let packings = match component {
    Fl2vaDenoiser | Ref2vaDenoiser =>
      @startup.denoiser_packings(plan, max_bytes=maximum_device_bytes)
    TextConditioner => @startup.conditioner_packings()
    VideoVae => @startup.video_decoder_packings(max_bytes=maximum_device_bytes)
    AudioVae => @startup.audio_decoder_packings(max_bytes=maximum_device_bytes)
  }
  @startup.component_bytes(component, packings[:], maximum_device_bytes)
  |> ignore
  packings
}

///|
/// Authenticate a component, bind every actual tensor to the closed model
/// schema and retain a directly consumable streaming manifest. Inventory order
/// remains the source ordinal throughout planning and upload.
pub fn ComponentCheckpoint::inspect(
  root : @approved_fs.ApprovedRoot,
  metadata : @h3.MiniMaxH3ModelMetadata,
  plan : @joint.JointDiffusionPlan,
  component : @weights.MiniMaxH3WeightComponent,
  inventory : @inventory.CheckpointInventory,
  maximum_component_bytes~ : Int64,
  maximum_device_bytes~ : Int64,
) -> Self raise {
  guard maximum_component_bytes > 0L && maximum_device_bytes > 0L else {
    raise @startup.StartupError::BudgetExceeded
  }
  let packings = component_packings(component, plan, maximum_device_bytes~)
  let bytes = @startup.component_bytes(
    component,
    packings[:],
    maximum_device_bytes,
  )
  let required = component.required_tensor_count()
  let streaming_limits = @streaming.StreamingSafetensorsLimits::new(
    maximum_shards=64,
    maximum_file_bytes=maximum_component_bytes,
    maximum_header_bytes=4194304,
    maximum_tensors_per_shard=required,
    maximum_total_tensors=required,
    maximum_rank=5,
    maximum_name_bytes=512,
    maximum_dtype_specs=2,
    authentication_chunk_bytes=1048576,
    maximum_copy_requests=required,
    maximum_copy_bytes=maximum_component_bytes,
  )
  let inspected = @streaming.inspect_shards(
    root,
    inventory.shards(),
    dtype_specs=[
      @streaming.StreamingSafetensorsDTypeSpec::bfloat16(),
      @streaming.StreamingSafetensorsDTypeSpec::float32(),
    ],
    limits=streaming_limits,
  )
  let header_limits = @tensor.HeaderLimits::new(
    max_header_bytes=4194304,
    max_tensor_count=required,
    max_rank=5,
    max_name_length=512,
    max_payload_bytes=maximum_component_bytes,
    max_materialized_tensor_bytes=maximum_component_bytes,
  )
  let partial : Array[@weights.MiniMaxH3PartialShardBinding] = []
  let declarations : Array[@host.MiniMaxH3ShardDeclarationInput] = []
  let claims : Array[@host.MiniMaxH3TensorIndexInput] = []
  let sources : Array[@host.MiniMaxH3ShardSource] = []
  for ordinal in 0..<inspected.shard_count() {
    guard inspected.shard(ordinal) is Some(shard) else {
      fail("inspected component shard missing")
    }
    let raw : Array[@tensor.RawTensorMetadata] = []
    for index in 0..<shard.tensor_count() {
      guard shard.tensor(index) is Some(t) else {
        fail("inspected tensor missing")
      }
      raw.push(
        @tensor.RawTensorMetadata::new(
          name=t.name(),
          dtype=t.dtype().tag(),
          shape=t.shape()[:],
          data_start=t.data_start(),
          data_end=t.data_end(),
        ),
      )
      claims.push(
        @host.MiniMaxH3TensorIndexInput::new(
          tensor_name=t.name(),
          shard_index=ordinal,
        ),
      )
    }
    let tensors = @tensor.validate_metadata(
      raw[:],
      encoded_header_bytes=(shard.payload_offset() - 8L).to_int(),
      payload_bytes=shard.payload_byte_count(),
      limits=header_limits,
    )
    partial.push(@weights.bind(metadata, plan, component, tensors))
    declarations.push(
      @host.MiniMaxH3ShardDeclarationInput::new(
        file_digest=@spec.ContentDigest::from_sha256(
          shard.file_digest().as_hex(),
        ),
        payload_bytes=shard.payload_byte_count(),
      ),
    )
    sources.push(
      @host.MiniMaxH3ShardSource::new(
        locator=inventory.shards()[ordinal].locator(),
      ),
    )
  }
  let binding = @weights.join_complete_component(component, partial[:])
  let limits = @host.MiniMaxH3HostMaterializationLimits::new(
    maximum_shards=64,
    maximum_tensors=required,
    maximum_header_bytes=4194304,
    maximum_payload_bytes_per_shard=maximum_component_bytes,
    maximum_total_host_bytes=maximum_component_bytes,
    maximum_arena_bytes=2147483584L,
    maximum_arenas=256,
    maximum_name_bytes=512,
  )
  let manifest = @host.build_manifest(
    binding,
    expected_identity=binding.identity(),
    shards=declarations[:],
    index=claims[:],
    limits~,
  )
  { manifest, sources, packings, limits, bytes, }
}

///|
pub fn ComponentCheckpoint::manifest(
  self : Self,
) -> @host.MiniMaxH3HostComponentManifest {
  self.manifest
}

///|
pub fn ComponentCheckpoint::required_device_bytes(self : Self) -> Int64 {
  self.bytes
}

///|
/// Upload into the caller's aggregate owner with bounded scratch and retained
/// cleanup authority on partial failure. The checkpoint root remains borrowed.
pub fn ComponentCheckpoint::prepare(
  self : Self,
  owner : @startup.WeightStartup,
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  chunk_bytes~ : Int,
) -> Unit raise {
  owner.prepare_component(
    context,
    root,
    self.sources[:],
    self.manifest,
    self.packings[:],
    expected_identity=self.manifest.identity(),
    expected_manifest_digest=self.manifest.digest(),
    limits=self.limits,
    chunk_bytes~,
  )
}
