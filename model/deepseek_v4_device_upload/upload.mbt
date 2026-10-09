///|
fn source_digest(
  weights : @deepseek_v4_host_materialization.DeepSeekV4HostWeights,
) -> @segmented_device_materialize.SegmentedHostSourceDigest raise DeepSeekV4DeviceUploadError {
  from_device(() => {
    @segmented_device_materialize.SegmentedHostSourceDigest::from_sha256(
      weights.manifest_digest().as_hex(),
    )
  })
}

///|
/// Build the deterministic family-neutral upload layout without touching a
/// device. Every DeepSeek tensor is one region in its final host arena.
pub fn plan_device_upload(
  weights : @deepseek_v4_host_materialization.DeepSeekV4HostWeights,
  limits : @segmented_device_materialize.SegmentedDeviceMaterializationLimits,
) -> @segmented_device_materialize.SegmentedDeviceLayout raise DeepSeekV4DeviceUploadError {
  if weights.is_released() {
    raise Host(ArenaReleased)
  }
  let arena_lengths : Array[Int] = []
  for arena_ordinal in 0..<weights.arena_count() {
    arena_lengths.push(
      from_borrow(() => weights.device_arena_size(arena_ordinal)),
    )
  }
  let regions : Array[
    @segmented_device_materialize.SegmentedDeviceTensorRegionInput,
  ] = []
  for tensor_ordinal in 0..<weights.tensor_count() {
    let tensor = from_host(() => weights.tensor(tensor_ordinal))
    regions.push(
      from_device(() => {
        @segmented_device_materialize.SegmentedDeviceTensorRegionInput::new(
          tensor_ordinal~,
          source_arena_ordinal=tensor.arena_ordinal(),
          source_offset=tensor.arena_offset(),
          byte_count=tensor.byte_count(),
        )
      }),
    )
  }
  let digest = source_digest(weights)
  from_device(() => {
    @segmented_device_materialize.plan_layout(
      source_identity=weights.identity(),
      source_digest=digest,
      source_arena_lengths=arena_lengths,
      regions~,
      limits~,
    )
  })
}

///|
/// Validate all opaque host borrows without allocating device memory.
pub fn validate_host_source(
  weights : @deepseek_v4_host_materialization.DeepSeekV4HostWeights,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> Unit raise DeepSeekV4DeviceUploadError {
  from_device(() => {
    @segmented_device_materialize.validate_borrowed_source(layout, provide=(
      arena_ordinal,
      consumer,
    ) => weights.provide_device_arena(arena_ordinal, consumer))
  })
}

///|
/// Upload authenticated raw host payloads into one explicitly releasable
/// device allocation. This does not make DeepSeek V4 executable.
pub fn materialize_device_weights(
  context : @device.Context,
  weights : @deepseek_v4_host_materialization.DeepSeekV4HostWeights,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> @segmented_device_materialize.SegmentedDevicePreparation raise DeepSeekV4DeviceUploadError {
  let digest = source_digest(weights)
  from_device(() => {
    @segmented_device_materialize.materialize_from_borrowed_source(
      context,
      layout,
      expected_identity=weights.identity(),
      expected_source_digest=digest,
      expected_layout_digest=layout.digest(),
      provide=(arena_ordinal, consumer) => {
        weights.provide_device_arena(arena_ordinal, consumer)
      },
    )
  })
}
