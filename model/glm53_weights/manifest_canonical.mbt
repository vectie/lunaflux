///|
fn hash_text(
  hasher : @canonical_sha256.CanonicalSha256,
  value : StringView,
) -> Unit {
  let bytes = @utf8.encode(value)
  ignore(hasher.update_u64_le(bytes.length().to_uint64()))
  ignore(hasher.update_bytes(bytes))
}

///|
fn manifest_digest(
  identity : @spec.ModelIdentity,
  encoding : Glm53WeightEncoding,
  tensors : ArrayView[Glm53TensorDescriptor],
  control_tokens : ArrayView[Int],
) -> @spec.PlanDigest {
  let hasher = @canonical_sha256.CanonicalSha256::new()
  hash_text(hasher, "lunaflux.glm53.tensor-manifest.v1")
  hash_text(hasher, identity.content().as_hex())
  hash_text(hasher, identity.plan().as_hex())
  ignore(
    hasher.update_byte(
      match encoding {
        BFloat16WithFloat32Routing => 0x01
        Fp8E4M3Block128Dynamic => 0x02
      },
    ),
  )
  ignore(hasher.update_u64_le(tensors.length().to_uint64()))
  for tensor in tensors {
    hash_text(hasher, tensor.name)
    ignore(
      hasher.update_byte(
        match tensor.dtype {
          BFloat16 => 0x01
          Float32 => 0x02
          Fp8E4M3 => 0x03
        },
      ),
    )
    ignore(hasher.update_u64_le(tensor.shape.length().to_uint64()))
    for dimension in tensor.shape {
      ignore(hasher.update_u64_le(dimension.reinterpret_as_uint64()))
    }
  }
  ignore(hasher.update_u64_le(control_tokens.length().to_uint64()))
  for token in control_tokens {
    ignore(hasher.update_u64_le(token.to_uint64()))
  }
  try! @spec.PlanDigest::from_sha256(hasher.finalize_hex())
}

///|
fn finish_manifest(
  profile : @glm53_spec.Glm53ProfileId,
  encoding : Glm53WeightEncoding,
  execution_identity : @spec.ModelIdentity,
  tensors : Array[Glm53TensorDescriptor],
  control_tokens : Array[Int],
) -> Glm53TensorManifest {
  let layout_digest = manifest_digest(
    execution_identity, encoding, tensors, control_tokens,
  )
  {
    profile,
    encoding,
    execution_identity,
    layout_digest,
    tensors: ReadOnlyArray::from_array(tensors),
    control_token_ids: ReadOnlyArray::from_array(control_tokens),
  }
}
