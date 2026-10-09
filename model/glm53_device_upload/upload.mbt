///|
fn source_digest(
  weights : @glm53_host_materialize.Glm53HostWeights,
) -> @segmented_device_materialize.SegmentedHostSourceDigest raise Glm53DeviceUploadError {
  from_device(() => {
    @segmented_device_materialize.SegmentedHostSourceDigest::from_sha256(
      weights.source_digest().as_hex(),
    )
  })
}

///|
/// Build a deterministic upload layout. A logical GLM tensor may contribute
/// several contiguous regions when host materialization split it across
/// bounded arenas; the generic layout retains its logical tensor ordinal.
pub fn plan_device_upload(
  weights : @glm53_host_materialize.Glm53HostWeights,
  limits : @segmented_device_materialize.SegmentedDeviceMaterializationLimits,
) -> @segmented_device_materialize.SegmentedDeviceLayout raise Glm53DeviceUploadError {
  if weights.is_closed() {
    raise Host(Closed)
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
    let tensor = from_host(() => weights.tensor_region(tensor_ordinal))
    for span_ordinal in 0..<tensor.span_count() {
      guard tensor.span(span_ordinal) is Some(span) else {
        raise Host(TensorOrdinalOutOfRange)
      }
      regions.push(
        from_device(() => {
          @segmented_device_materialize.SegmentedDeviceTensorRegionInput::new(
            tensor_ordinal~,
            source_arena_ordinal=span.arena_index(),
            source_offset=span.offset_bytes(),
            byte_count=span.byte_length(),
          )
        }),
      )
    }
  }
  let digest = source_digest(weights)
  from_device(() => {
    @segmented_device_materialize.plan_layout(
      source_identity=weights.execution_identity(),
      source_digest=digest,
      source_arena_lengths=arena_lengths,
      regions~,
      limits~,
    )
  })
}

///|
pub fn validate_host_source(
  weights : @glm53_host_materialize.Glm53HostWeights,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> Unit raise Glm53DeviceUploadError {
  from_device(() => {
    @segmented_device_materialize.validate_borrowed_source(layout, provide=(
      arena_ordinal,
      consumer,
    ) => weights.provide_device_arena(arena_ordinal, consumer))
  })
}

///|
/// Upload authenticated raw host payloads into one explicitly releasable
/// device allocation. Kernel qualification remains a separate admission.
pub fn materialize_device_weights(
  context : @device.Context,
  weights : @glm53_host_materialize.Glm53HostWeights,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> @segmented_device_materialize.SegmentedDevicePreparation raise Glm53DeviceUploadError {
  let digest = source_digest(weights)
  from_device(() => {
    @segmented_device_materialize.materialize_from_borrowed_source(
      context,
      layout,
      expected_identity=weights.execution_identity(),
      expected_source_digest=digest,
      expected_layout_digest=layout.digest(),
      provide=(arena_ordinal, consumer) => {
        weights.provide_device_arena(arena_ordinal, consumer)
      },
    )
  })
}
