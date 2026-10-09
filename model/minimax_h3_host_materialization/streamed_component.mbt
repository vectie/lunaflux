///|
/// Startup metadata and tensor locations only; no host weight arenas are held.
pub struct MiniMaxH3StreamingComponent {
  priv manifest : MiniMaxH3HostComponentManifest
  priv admission : MiniMaxH3StreamingAdmission
  priv limits : @streaming_safetensors.StreamingSafetensorsLimits
}

///|
pub fn MiniMaxH3StreamingComponent::component(
  self : Self,
) -> @minimax_h3_weights.MiniMaxH3WeightComponent {
  self.manifest.component
}

///|
pub fn inspect_streaming_component(
  root : @approved_fs.ApprovedRoot,
  sources : ArrayView[MiniMaxH3ShardSource],
  manifest : MiniMaxH3HostComponentManifest,
  expected_identity~ : @spec.ModelIdentity,
  expected_manifest_digest~ : MiniMaxH3HostManifestDigest,
  limits~ : MiniMaxH3HostMaterializationLimits,
) -> MiniMaxH3StreamingComponent raise MiniMaxH3HostMaterializationError {
  validate_materialization_admission(
    manifest, expected_identity, expected_manifest_digest, limits,
  )
  {
    manifest,
    admission: inspect_component_shards(root, sources, manifest, limits),
    limits: streaming_limits(limits),
  }
}

///|
pub fn MiniMaxH3StreamingComponent::manifest(
  self : Self,
) -> MiniMaxH3HostComponentManifest {
  self.manifest
}

///|
/// Pure translation of the model's existing packing into checkpoint slices.
/// Device ownership and scratch capacity are deliberately not model semantics.
pub fn MiniMaxH3StreamingComponent::transfers(
  self : Self,
  copies : ArrayView[@minimax_h3_weights.BlockWeightCopy],
  destination_ordinal~ : Int,
) -> Array[@streaming_safetensors.StreamingSafetensorsTransfer] raise MiniMaxH3HostMaterializationError {
  copies.map(copy => {
    let mut index = -1
    for i, tensor in self.manifest.tensors {
      if tensor.tensor.name() == copy.tensor_name {
        index = i
        break
      }
    }
    guard index >= 0 else { raise StreamingTensorMismatch(-1) }
    let tensor = self.manifest.tensors[index].tensor
    guard copy.source_offset >= 0L &&
      copy.byte_count > 0L &&
      copy.byte_count <= tensor.byte_length() &&
      copy.source_offset <= tensor.byte_length() - copy.byte_count else {
      raise TensorByteOutsideArena(index)
    }
    let location = self.admission.locations[index]
    @streaming_safetensors.StreamingSafetensorsTransfer::new(
      shard_ordinal=location.shard_ordinal,
      tensor_ordinal=location.tensor_ordinal,
      tensor_byte_offset=copy.source_offset,
      destination_ordinal~,
      destination_offset=copy.destination_offset,
      byte_count=copy.byte_count,
    ) catch {
      _ => raise StreamingFailure
    }
  })
}

///|
/// Startup-only bounded upload. The consumer must synchronously consume each
/// scratch chunk. Failure leaves destination ownership with the caller.
pub fn MiniMaxH3StreamingComponent::transfer(
  self : Self,
  root : @approved_fs.ApprovedRoot,
  requests : ArrayView[@streaming_safetensors.StreamingSafetensorsTransfer],
  destination_bytes~ : ArrayView[Int64],
  scratch~ : FixedArray[Byte],
  consume~ : (Int, Int64, FixedArray[Byte], Int) -> Bool,
) -> Unit raise MiniMaxH3HostMaterializationError {
  @streaming_safetensors.transfer_ranges(
    root,
    self.admission.inspection,
    requests~,
    destination_bytes~,
    scratch~,
    limits=self.limits,
    consume~,
  ) catch {
    _ => raise StreamingFailure
  }
}
