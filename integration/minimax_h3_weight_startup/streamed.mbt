///|
fn Packing::stream_copies(
  self : Self,
) -> Array[@weights.BlockWeightCopy] raise StartupError {
  match self {
    Block(p) => p.copies[:].to_owned()
    Input(p) => p.copies[:].to_owned()
    Timestep(p) => p.copies[:].to_owned()
    Output(p) => p.copies[:].to_owned()
    Context(p) => p.copies[:].to_owned()
    Refiner(p) => p.copies[:].to_owned()
    VideoFrontend(p) => p.copies[:].to_owned()
    VideoBlock(p) => p.copies[:].to_owned()
    AudioDecoder(p) => p.copies[:].to_owned()
    AudioAmp(p) =>
      p.regions()[:].map(r => {
        @weights.BlockWeightCopy::new(
          tensor_name=r.tensor_name(),
          source_offset=0L,
          destination_offset=r.byte_offset(),
          byte_count=r.byte_count(),
        ) catch {
          _ => raise InvalidPacking
        }
      })
    _ => raise InvalidPacking
  }
}

///|
fn WeightStartup::prepare_streamed_packings(
  self : Self,
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  sources : ArrayView[@host.MiniMaxH3ShardSource],
  manifest : @host.MiniMaxH3HostComponentManifest,
  packings : ArrayView[Packing],
  expected_identity : @spec.ModelIdentity,
  expected_manifest_digest : @host.MiniMaxH3HostManifestDigest,
  limits : @host.MiniMaxH3HostMaterializationLimits,
  chunk_bytes : Int,
) -> Unit raise {
  let component = @host.inspect_streaming_component(
    root,
    sources,
    manifest,
    expected_identity~,
    expected_manifest_digest~,
    limits~,
  )
  let requests : Array[@streaming.StreamingSafetensorsTransfer] = []
  let sizes : Array[Int64] = []
  for packing in packings {
    if packing is (TextEncoder | VisionConditioner | ReferenceEncoder(_, _)) {
      continue
    }
    let i = sizes.length()
    sizes.push(packing.bytes())
    let copies = packing.stream_copies()
    requests.append(component.transfers(copies[:], destination_ordinal=i))
  }
  let allocations : Array[@device.Allocation] = []
  for packing in packings {
    let bytes = packing.bytes()
    match packing {
      TextEncoder => {
        let owner = @text.EncoderWeights::new()
        self.entries.push({
          owner: Encoder(owner),
          bytes,
          component: manifest.component(),
        })
        self.used += bytes
        owner.prepare_streamed(
          context,
          root,
          component,
          max_bytes=bytes,
          chunk_bytes~,
        )
      }
      VisionConditioner => {
        let owner = @vision.VisionWeights::new()
        self.entries.push({
          owner: Visual(owner),
          bytes,
          component: manifest.component(),
        })
        self.used += bytes
        owner.prepare_streamed(
          context,
          root,
          component,
          max_bytes=bytes,
          chunk_bytes~,
        )
      }
      ReferenceEncoder(limits, _) => {
        let metadata = @media.WeightManifest::from_manifest(manifest)
        let layout = @component_upload.plan_manifest_upload(manifest, limits)
        guard layout.device_arena_bytes() <= bytes else { raise BudgetExceeded }
        match
          @component_upload.materialize_streamed_weights(
            context,
            root,
            component,
            layout,
            chunk_bytes~,
          ) {
          SegmentedDeviceWeightsReady(owner) => {
            self.entries.push({
              owner: Segmented(owner, metadata),
              bytes,
              component: manifest.component(),
            })
            self.used += bytes
          }
          SegmentedDeviceCleanupRequired(owner) => {
            self.entries.push({
              owner: Cleanup(owner),
              bytes,
              component: manifest.component(),
            })
            self.used += bytes
            raise owner.cause()
          }
        }
      }
      _ => {
        let allocation = context.allocate(bytes)
        self.entries.push({
          owner: Packed(allocation),
          bytes,
          component: manifest.component(),
        })
        self.used += bytes
        allocations.push(allocation)
      }
    }
  }
  if !requests.is_empty() {
    @upload.upload_streamed(
      context,
      root,
      component,
      allocations[:],
      requests[:],
      destination_bytes=sizes[:],
      chunk_bytes~,
    )
  }
}
