///|
/// Materialize exactly one admitted component, upload every requested packed
/// owner, then zeroize/drop the host arena before returning. In particular text
/// and visual weights sharing one TextConditioner manifest use one host load.
/// Host arenas are GC managed: release drops references, not an OS RSS promise.
pub fn WeightStartup::prepare_component(
  self : Self,
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  sources : ArrayView[@host.MiniMaxH3ShardSource],
  manifest : @host.MiniMaxH3HostComponentManifest,
  packings : ArrayView[Packing],
  expected_identity~ : @spec.ModelIdentity,
  expected_manifest_digest~ : @host.MiniMaxH3HostManifestDigest,
  limits~ : @host.MiniMaxH3HostMaterializationLimits,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard !self.failed && !self.closed && self.leases == 0 && chunk_bytes > 0 else {
    raise InvalidState
  }
  let component = manifest.component()
  guard !self.completed.contains(component) else { raise InvalidState }
  if self.identity is Some(identity) {
    guard identity == expected_identity else { raise InvalidState }
  }
  component_bytes(component, packings, self.maximum - self.used) |> ignore
  self.identity = Some(expected_identity)
  self.failed = true
  let arena = @host.materialize_safetensors_shards(
    root,
    sources,
    manifest,
    expected_identity~,
    expected_manifest_digest~,
    limits~,
  )
  try {
    for packing in packings {
      let bytes = packing.bytes()
      match packing {
        TextEncoder => {
          let owner = @text.EncoderWeights::new()
          self.entries.push({ owner: Encoder(owner), bytes, component, })
          self.used += bytes
          owner.prepare(
            context,
            arena,
            expected_identity~,
            max_bytes=bytes,
            chunk_bytes~,
          )
        }
        VisionConditioner => {
          let owner = @vision.VisionWeights::new()
          self.entries.push({ owner: Visual(owner), bytes, component, })
          self.used += bytes
          owner.prepare(
            context,
            arena,
            expected_identity~,
            max_bytes=bytes,
            chunk_bytes~,
          )
        }
        ReferenceEncoder(limits, _) => {
          let metadata = @media.WeightManifest::new(arena)
          let layout = @component_upload.plan_device_upload(arena, limits)
          guard layout.device_arena_bytes() <= bytes else {
            raise BudgetExceeded
          }
          let preparation = @component_upload.materialize_device_weights(
            context, arena, layout,
          )
          match preparation {
            SegmentedDeviceWeightsReady(owner) => {
              self.entries.push({
                owner: Segmented(owner, metadata),
                bytes,
                component,
              })
              self.used += bytes
            }
            SegmentedDeviceCleanupRequired(owner) => {
              self.entries.push({ owner: Cleanup(owner), bytes, component, })
              self.used += bytes
              raise owner.cause()
            }
          }
        }
        _ => {
          let allocation = context.allocate(bytes)
          self.entries.push({ owner: Packed(allocation), bytes, component, })
          self.used += bytes
          upload_packed(
            context,
            allocation,
            arena,
            packing,
            expected_identity~,
            chunk_bytes~,
          )
        }
      }
    }
  } catch {
    error => {
      arena.release() |> ignore
      raise error
    }
  }
  arena.release() |> ignore
  self.completed.push(component)
  self.failed = false
}

///|
fn upload_packed(
  context : @device.Context,
  allocation : @device.Allocation,
  arena : @host.MiniMaxH3HostComponentArena,
  packing : Packing,
  expected_identity~ : @spec.ModelIdentity,
  chunk_bytes~ : Int,
) -> Unit raise {
  match packing {
    Block(p) =>
      @upload.upload_block(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        destination_offset=0L,
        chunk_bytes~,
      )
    Input(p) =>
      @upload.upload_input(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        destination_offset=0L,
        chunk_bytes~,
      )
    Timestep(p) =>
      @upload.upload_timestep(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        destination_offset=0L,
        chunk_bytes~,
      )
    Output(p) =>
      @upload.upload_output(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        destination_offset=0L,
        chunk_bytes~,
      )
    Context(p) =>
      @upload.upload_context(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        destination_offset=0L,
        chunk_bytes~,
      )
    Refiner(p) =>
      @upload.upload_refiner(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        destination_offset=0L,
        chunk_bytes~,
      )
    VideoFrontend(p) =>
      @upload.upload_video_frontend(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        chunk_bytes~,
      )
    VideoBlock(p) =>
      @upload.upload_video_block(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        chunk_bytes~,
      )
    AudioDecoder(p) =>
      @upload.upload_audio_decoder(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        chunk_bytes~,
      )
    AudioAmp(p) =>
      @upload.upload_audio_amp(
        context,
        allocation,
        arena,
        p,
        expected_identity~,
        destination_offset=0L,
        chunk_bytes~,
      )
    TextEncoder | VisionConditioner | ReferenceEncoder(_, _) =>
      raise InvalidPacking
  }
}
