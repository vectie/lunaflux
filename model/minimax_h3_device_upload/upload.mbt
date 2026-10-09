///|
fn source_digest(
  arena : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
) -> @segmented_device_materialize.SegmentedHostSourceDigest raise MiniMaxH3DeviceUploadError {
  from_device(() => {
    @segmented_device_materialize.SegmentedHostSourceDigest::from_sha256(
      arena.manifest_digest().as_hex(),
    )
  })
}

///|
/// Build the deterministic family-neutral upload layout for one admitted
/// MiniMax H3 component. Denoiser and conditioner stay separate owners.
pub fn plan_device_upload(
  arena : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  limits : @segmented_device_materialize.SegmentedDeviceMaterializationLimits,
) -> @segmented_device_materialize.SegmentedDeviceLayout raise MiniMaxH3DeviceUploadError {
  if arena.is_released() {
    raise Host(ArenaReleased)
  }
  let arena_lengths : Array[Int] = []
  for arena_ordinal in 0..<arena.arena_count() {
    arena_lengths.push(
      from_borrow(() => arena.device_arena_size(arena_ordinal)),
    )
  }
  let regions : Array[
    @segmented_device_materialize.SegmentedDeviceTensorRegionInput,
  ] = []
  for tensor_ordinal in 0..<arena.tensor_count() {
    let tensor = from_host(() => arena.tensor(tensor_ordinal))
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
  let digest = source_digest(arena)
  from_device(() => {
    @segmented_device_materialize.plan_layout(
      source_identity=arena.identity(),
      source_digest=digest,
      source_arena_lengths=arena_lengths,
      regions~,
      limits~,
    )
  })
}

///|
pub fn validate_host_source(
  arena : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> Unit raise MiniMaxH3DeviceUploadError {
  from_device(() => {
    @segmented_device_materialize.validate_borrowed_source(layout, provide=(
      arena_ordinal,
      consumer,
    ) => arena.provide_device_arena(arena_ordinal, consumer))
  })
}

///|
/// Upload one component into an explicitly releasable device allocation. This
/// does not erase the typed missing-VAE limitation or grant execution.
pub fn materialize_device_weights(
  context : @device.Context,
  arena : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> @segmented_device_materialize.SegmentedDevicePreparation raise MiniMaxH3DeviceUploadError {
  let digest = source_digest(arena)
  from_device(() => {
    @segmented_device_materialize.materialize_from_borrowed_source(
      context,
      layout,
      expected_identity=arena.identity(),
      expected_source_digest=digest,
      expected_layout_digest=layout.digest(),
      provide=(arena_ordinal, consumer) => {
        arena.provide_device_arena(arena_ordinal, consumer)
      },
    )
  })
}
