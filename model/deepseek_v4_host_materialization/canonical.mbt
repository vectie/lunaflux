///|
fn hash_text(hash : @crypto.SHA256, value : StringView) -> Unit {
  let prefix = @utf8.encode(value.length().to_string() + ":")
  let encoded = @utf8.encode(value)
  hash.update_from_iter(prefix.iter())
  hash.update_from_iter(encoded.iter())
  hash.update_from_iter([b'\n'].iter())
}

///|
fn dtype_canonical(
  dtype : @deepseek_v4_weights.DeepSeekV4SafetensorsDType,
) -> String {
  match dtype {
    BF16 => "BF16"
    F32 => "F32"
    I64 => "I64"
    I8 => "I8"
    F8E4M3 => "F8_E4M3"
    F8E8M0 => "F8_E8M0"
  }
}

///|
fn encoding_canonical(
  encoding : @deepseek_v4_weights.DeepSeekV4WeightEncoding,
) -> String {
  match encoding {
    BFloat16 => "bfloat16"
    Float32 => "float32"
    Int64 => "int64"
    Fp8E4M3Block128 => "fp8-e4m3-block128"
    Ue8M0Scale => "ue8m0-scale"
    PackedFp4E2M1Block32 => "packed-fp4-e2m1-block32"
  }
}

///|
fn canonical_manifest_digest(
  identity : @spec.ModelIdentity,
  expected : DeepSeekV4HostExpectedDigests,
  binding : @deepseek_v4_weights.DeepSeekV4WeightBinding,
  inspection : @streaming_safetensors.StreamingSafetensorsInspection,
  layouts : ReadOnlyArray[DeepSeekV4HostLayoutTensor],
  arena_sizes : ReadOnlyArray[Int],
  total_tensor_bytes : Int64,
  total_arena_bytes : Int64,
) -> DeepSeekV4HostManifestDigest raise DeepSeekV4HostMaterializationError {
  let hash = @crypto.SHA256::new()
  hash_text(hash, "lunaflux/deepseek-v4-host-manifest/v1")
  hash_text(hash, identity.content().as_hex())
  hash_text(hash, identity.plan().as_hex())
  hash_text(hash, expected.logical_manifest_digest.as_hex())
  hash_text(hash, expected.numeric_schema_digest.as_hex())
  hash_text(hash, expected.numeric_binding_digest.as_hex())
  hash_text(hash, inspection.shard_count().to_string())
  for shard_ordinal in 0..<inspection.shard_count() {
    guard inspection.shard(shard_ordinal) is Some(shard) else {
      raise StreamingFailure
    }
    hash_text(hash, shard_ordinal.to_string())
    hash_text(hash, shard.file_digest().as_hex())
    hash_text(hash, shard.file_byte_count().to_string())
    hash_text(hash, shard.payload_offset().to_string())
    hash_text(hash, shard.payload_byte_count().to_string())
  }
  hash_text(hash, layouts.length().to_string())
  for layout in layouts {
    guard binding.tensor(layout.binding_index) is Some(bound) else {
      raise WeightBindingFailure
    }
    let tensor = bound.tensor()
    hash_text(hash, layout.binding_index.to_string())
    hash_text(hash, tensor.name())
    hash_text(hash, dtype_canonical(tensor.dtype()))
    hash_text(hash, encoding_canonical(bound.encoding()))
    hash_text(hash, tensor.shape().length().to_string())
    for dimension in tensor.shape() {
      hash_text(hash, dimension.to_string())
    }
    hash_text(hash, layout.shard_ordinal.to_string())
    hash_text(hash, layout.tensor_ordinal.to_string())
    hash_text(hash, tensor.data_start().to_string())
    hash_text(hash, tensor.data_end().to_string())
    hash_text(hash, layout.arena_ordinal.to_string())
    hash_text(hash, layout.arena_offset.to_string())
    hash_text(hash, layout.byte_count.to_string())
    hash_text(
      hash,
      match layout.quantized_parameter_index {
        Some(index) => index.to_string()
        None => "none"
      },
    )
  }
  hash_text(hash, arena_sizes.length().to_string())
  for size in arena_sizes {
    hash_text(hash, size.to_string())
  }
  hash_text(hash, total_tensor_bytes.to_string())
  hash_text(hash, total_arena_bytes.to_string())
  { hex: @crypto.bytes_to_hex_string(hash.finalize()), }
}
