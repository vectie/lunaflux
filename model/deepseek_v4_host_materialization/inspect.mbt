///|
fn deepseek_dtype_specs() -> Array[
  @streaming_safetensors.StreamingSafetensorsDTypeSpec,
] raise DeepSeekV4HostMaterializationError {
  let specs = [
    @streaming_safetensors.StreamingSafetensorsDTypeSpec::bfloat16(),
    @streaming_safetensors.StreamingSafetensorsDTypeSpec::float32(),
  ]
  for
    tag_and_width in [("I64", 8L), ("I8", 1L), ("F8_E4M3", 1L), ("F8_E8M0", 1L)] {
    let (tag, width) = tag_and_width
    specs.push(
      @streaming_safetensors.StreamingSafetensorsDTypeSpec::opaque_fixed_width(
        tag,
        bytes_per_element=width,
      ) catch {
        _ => raise StreamingFailure
      },
    )
  }
  specs
}

///|
fn validate_index_membership(
  inspection : @streaming_safetensors.StreamingSafetensorsInspection,
  index : ArrayView[DeepSeekV4TensorIndexInput],
  maximum_entries : Int,
) -> Unit raise DeepSeekV4HostMaterializationError {
  let expected = inspection.total_tensor_count()
  if index.length() != expected || index.length() > maximum_entries {
    raise InvalidIndexCount(expected~, actual=index.length())
  }
  let claims : Map[String, (Int, Int)] = Map([], capacity=index.length())
  for claim_index, claim in index {
    if claim.tensor_name.is_empty() ||
      claim.tensor_name.length() > 256 ||
      claim.shard_ordinal < 0 ||
      claim.shard_ordinal >= inspection.shard_count() {
      raise InvalidIndexEntry(claim_index)
    }
    match claims.get(claim.tensor_name) {
      Some((previous, _)) => raise DuplicateIndexEntry(previous, claim_index)
      None => claims.set(claim.tensor_name, (claim_index, claim.shard_ordinal))
    }
  }
  let mut observed = 0
  for shard_ordinal in 0..<inspection.shard_count() {
    guard inspection.shard(shard_ordinal) is Some(shard) else {
      raise StreamingFailure
    }
    for tensor_ordinal in 0..<shard.tensor_count() {
      guard shard.tensor(tensor_ordinal) is Some(tensor) else {
        raise StreamingFailure
      }
      match claims.get(tensor.name()) {
        Some((_, claimed_shard)) if claimed_shard == shard_ordinal => ()
        Some(_) => raise IndexShardMismatch(observed)
        None => raise InvalidIndexEntry(observed)
      }
      observed += 1
    }
  }
  if observed != expected {
    raise InvalidIndexCount(expected~, actual=observed)
  }
}

///|
fn bind_authenticated_tensors(
  plan : @deepseek_v4.DeepSeekV4SemanticPlan,
  inspection : @streaming_safetensors.StreamingSafetensorsInspection,
) -> (
  @deepseek_v4_weights.DeepSeekV4WeightBinding,
  Array[HostLayoutSourceTensor],
) raise DeepSeekV4HostMaterializationError {
  let tensors : Array[@deepseek_v4_weights.DeepSeekV4SafetensorsTensor] = []
  let sources : Array[HostLayoutSourceTensor] = []
  for shard_ordinal in 0..<inspection.shard_count() {
    guard inspection.shard(shard_ordinal) is Some(shard) else {
      raise StreamingFailure
    }
    for tensor_ordinal in 0..<shard.tensor_count() {
      guard shard.tensor(tensor_ordinal) is Some(tensor) else {
        raise StreamingFailure
      }
      let bound = @deepseek_v4_weights.DeepSeekV4SafetensorsTensor::new(
        name=tensor.name(),
        dtype=tensor.dtype().tag(),
        shape=tensor.shape(),
        shard=shard_ordinal,
        data_start=tensor.data_start(),
        data_end=tensor.data_end(),
      ) catch {
        _ => raise WeightBindingFailure
      }
      tensors.push(bound)
      sources.push({
        shard_ordinal,
        tensor_ordinal,
        byte_count: tensor.byte_count(),
      })
    }
  }
  let binding = @deepseek_v4_weights.bind(plan, tensors) catch {
    _ => raise WeightBindingFailure
  }
  (binding, sources)
}

///|
fn validate_numeric_identity(
  plan : @deepseek_v4.DeepSeekV4SemanticPlan,
  binding : @deepseek_v4_weights.DeepSeekV4WeightBinding,
  expected : DeepSeekV4HostExpectedDigests,
) -> @deepseek_v4_numeric_plan.DeepSeekV4NumericPlan raise DeepSeekV4HostMaterializationError {
  let numeric = @deepseek_v4_numeric_plan.build(plan) catch {
    _ => raise NumericPlanFailure
  }
  numeric.validate_weight_manifest(binding) catch {
    _ => raise NumericPlanFailure
  }
  if binding.identity() != expected.identity ||
    numeric.identity() != expected.identity {
    raise IdentityMismatch
  }
  if numeric.logical_manifest_digest() != expected.logical_manifest_digest {
    raise LogicalManifestDigestMismatch
  }
  if numeric.numeric_schema_digest() != expected.numeric_schema_digest {
    raise NumericSchemaDigestMismatch
  }
  if numeric.binding_digest() != expected.numeric_binding_digest {
    raise NumericBindingDigestMismatch
  }
  numeric
}
