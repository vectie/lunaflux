///|
pub(all) suberror PackedExpertUploadError {
  InvalidChunkSize
  InvalidDestinationCount
}

///|
/// Bind a per-device expert partition to the original packed checkpoint.
/// Model names/shapes come from the family adapter, storage/streaming from the
/// shared precision path. Returned order is expert-major gate/up/down.
pub fn bind_experts(
  checkpoint : @precision_materialize.PackedCheckpoint,
  layer_index~ : Int,
  expert_indices~ : ArrayView[Int],
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  @glm53_weights.nvfp4_expert_weights(layer_index~, expert_indices~).map(weight => {
    checkpoint.bind_weight(
      representation=weight.representation(),
      payload_name=weight.payload_name(),
      scales_name=weight.scales_name(),
      global_scale_name=weight.global_scale_name(),
    )
  })
}

///|
/// Startup upload directly into caller-owned device allocations. Each buffer
/// holds packed payload, block scales, and global scale at the binding offsets.
/// No full host bank or BF16 duplicate is created. The caller releases all
/// allocations on a failed upload and publishes only after this call succeeds.
pub fn upload_experts(
  context : @device.Context,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  weights : ArrayView[@precision_materialize.PackedCheckpointWeight],
  allocations : ArrayView[@device.Allocation],
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  guard allocations.length() == weights.length() else {
    raise InvalidDestinationCount
  }
  for i, allocation in allocations {
    context.validate_allocation_region(
      allocation,
      offset=0L,
      byte_count=weights[i].buffer_bytes(),
      alignment=4L,
    )
  }
  checkpoint.transfer_weights(
    root,
    weights,
    scratch=FixedArray::make(chunk_bytes, b'\x00'),
    limits~,
    consume=(destination, offset, buffer, count) => {
      try
        allocations[destination].copy_from_host(
          Bytes::from_array(buffer[:count]),
          0L,
          offset,
          count.to_int64(),
        )
      catch {
        _ => false
      } noraise {
        _ => true
      }
    },
  )
}
