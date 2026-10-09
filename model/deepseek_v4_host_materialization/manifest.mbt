///|
/// Authenticate exact shards, bind every header tensor to the closed DeepSeek
/// V4 schema, join numeric identity, and derive a bounded final-arena layout.
pub fn build_manifest(
  root : @approved_fs.ApprovedRoot,
  plan : @deepseek_v4.DeepSeekV4SemanticPlan,
  shards : ArrayView[@streaming_safetensors.StreamingSafetensorsShardInput],
  index~ : ArrayView[DeepSeekV4TensorIndexInput],
  expected_digests~ : DeepSeekV4HostExpectedDigests,
  streaming_limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  limits~ : DeepSeekV4HostMaterializationLimits,
) -> DeepSeekV4HostManifest raise DeepSeekV4HostMaterializationError {
  let inspection = @streaming_safetensors.inspect_shards(
    root,
    shards,
    dtype_specs=deepseek_dtype_specs(),
    limits=streaming_limits,
  ) catch {
    _ => raise StreamingFailure
  }
  validate_index_membership(inspection, index, limits.maximum_index_entries)
  let (binding, sources) = bind_authenticated_tensors(plan, inspection)
  let numeric_plan = validate_numeric_identity(plan, binding, expected_digests)
  let (unpaired, arena_sizes, total_tensor_bytes, total_arena_bytes) = build_layout(
    sources, limits,
  )
  let tensors = attach_quantized_parameter_pairs(binding, unpaired)
  if binding.tensor_count() != tensors.length() {
    raise WeightBindingFailure
  }
  let identity = binding.identity()
  let digest = canonical_manifest_digest(
    identity, expected_digests, binding, inspection, tensors, arena_sizes, total_tensor_bytes,
    total_arena_bytes,
  )
  {
    identity,
    expected_digests,
    digest,
    binding,
    numeric_plan,
    inspection,
    tensors,
    arena_sizes,
    total_tensor_bytes,
    total_arena_bytes,
    streaming_limits,
  }
}
