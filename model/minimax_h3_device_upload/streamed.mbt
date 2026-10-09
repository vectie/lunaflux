///|
/// Preserve the reference encoder's complete tensor table and device offsets,
/// but derive the layout from metadata rather than resident host arenas.
pub fn plan_manifest_upload(
  manifest : @minimax_h3_host_materialization.MiniMaxH3HostComponentManifest,
  limits : @segmented_device_materialize.SegmentedDeviceMaterializationLimits,
) -> @segmented_device_materialize.SegmentedDeviceLayout raise MiniMaxH3DeviceUploadError {
  let lengths : Array[Int] = []
  for i in 0..<manifest.arena_count() {
    guard manifest.arena_size(i) is Some(bytes) else {
      raise Host(ArenaReleased)
    }
    lengths.push(bytes.to_int())
  }
  let regions : Array[
    @segmented_device_materialize.SegmentedDeviceTensorRegionInput,
  ] = []
  for tensor_ordinal in 0..<manifest.tensor_count() {
    let tensor = from_host(() => manifest.tensor(tensor_ordinal))
    regions.push(
      from_device(() => {
        @segmented_device_materialize.SegmentedDeviceTensorRegionInput::new(
          tensor_ordinal~,
          source_arena_ordinal=tensor.arena_index(),
          source_offset=tensor.arena_offset().to_int(),
          byte_count=tensor.byte_length().to_int(),
        )
      }),
    )
  }
  from_device(() => {
    @segmented_device_materialize.plan_layout(
      source_identity=manifest.identity(),
      source_digest=@segmented_device_materialize.SegmentedHostSourceDigest::from_sha256(
        manifest.digest().as_hex(),
      ),
      source_arena_lengths=lengths[:],
      regions~,
      limits~,
    )
  })
}

///|
pub fn materialize_streamed_weights(
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  component : @minimax_h3_host_materialization.MiniMaxH3StreamingComponent,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
  chunk_bytes~ : Int,
) -> @segmented_device_materialize.SegmentedDevicePreparation raise MiniMaxH3DeviceUploadError {
  guard chunk_bytes > 0 else { raise Device(InvalidLimit) }
  let manifest = component.manifest()
  guard layout.source_identity() == manifest.identity() &&
    layout.source_digest().as_hex() == manifest.digest().as_hex() &&
    layout.region_count() == manifest.tensor_count() else {
    raise Device(SourceDigestMismatch)
  }
  let copies : Array[@minimax_h3_weights.BlockWeightCopy] = []
  for i in 0..<layout.region_count() {
    guard layout.region(i) is Some(region) else {
      raise Device(InvalidTensorOrdinal(i))
    }
    let tensor = from_host(() => manifest.tensor(region.tensor_ordinal()))
    guard region.source_arena_ordinal() == tensor.arena_index() &&
      region.source_offset().to_int64() == tensor.arena_offset() &&
      region.byte_count() == tensor.byte_length() else {
      raise Device(InvalidSourceRegion(i))
    }
    copies.push(
      @minimax_h3_weights.BlockWeightCopy::new(
        tensor_name=tensor.name(),
        source_offset=0L,
        destination_offset=region.device_offset(),
        byte_count=region.byte_count(),
      ) catch {
        _ => raise Host(StreamingFailure)
      },
    )
  }
  let requests = from_host(() => {
    component.transfers(copies[:], destination_ordinal=0)
  })
  from_device(() => {
    @segmented_device_materialize.materialize_from_streamed_source(
      context,
      layout,
      upload=allocation => {
        component.transfer(
          root,
          requests[:],
          destination_bytes=[layout.device_arena_bytes()],
          scratch=FixedArray::make(chunk_bytes, b'\x00'),
          consume=(_, offset, scratch, count) => {
            let bytes = Bytes::from_array(scratch[:count])
            try
              allocation.copy_from_host(bytes, 0L, offset, count.to_int64())
            catch {
              _ => false
            } noraise {
              _ => true
            }
          },
        ) catch {
          _ => raise HostSourceUnavailable(-1)
        }
      },
    )
  })
}
