///|
pub fn DeepSeekV4HostMaterializationLimits::new(
  maximum_total_host_bytes~ : Int64,
  maximum_arena_bytes~ : Int,
  maximum_arenas~ : Int,
  maximum_index_entries~ : Int,
) -> Self raise DeepSeekV4HostMaterializationError {
  if maximum_total_host_bytes <= 0L ||
    maximum_arena_bytes <= 0 ||
    maximum_arena_bytes > 2147483647 ||
    maximum_arena_bytes.to_int64() > maximum_total_host_bytes ||
    maximum_arenas <= 0 ||
    maximum_arenas > 4096 ||
    maximum_index_entries <= 0 ||
    maximum_index_entries > 145118 {
    raise InvalidLimit
  }
  {
    maximum_total_host_bytes,
    maximum_arena_bytes,
    maximum_arenas,
    maximum_index_entries,
  }
}

///|
pub fn DeepSeekV4HostExpectedDigests::new(
  identity~ : @spec.ModelIdentity,
  logical_manifest_digest~ : @spec.PlanDigest,
  numeric_schema_digest~ : @spec.PlanDigest,
  numeric_binding_digest~ : @spec.PlanDigest,
) -> Self {
  {
    identity,
    logical_manifest_digest,
    numeric_schema_digest,
    numeric_binding_digest,
  }
}

///|
pub fn DeepSeekV4TensorIndexInput::new(
  tensor_name~ : String,
  shard_ordinal~ : Int,
) -> Self {
  { tensor_name, shard_ordinal, }
}

///|
pub fn DeepSeekV4HostExpectedDigests::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeepSeekV4HostExpectedDigests::logical_manifest_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.logical_manifest_digest
}

///|
pub fn DeepSeekV4HostExpectedDigests::numeric_schema_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.numeric_schema_digest
}

///|
pub fn DeepSeekV4HostExpectedDigests::numeric_binding_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.numeric_binding_digest
}

///|
pub fn DeepSeekV4HostManifestDigest::as_hex(self : Self) -> String {
  self.hex
}
