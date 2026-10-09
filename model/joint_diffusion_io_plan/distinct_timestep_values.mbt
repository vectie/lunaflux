///|
pub fn JointDistinctTimestepValues::new(
  request : JointDiffusionIoPlan,
  packed_timestep : JointPackedTimestepSelection,
  values~ : ArrayView[Float],
) -> Self raise JointDiffusionIoPlanError {
  if packed_timestep.identity() != request.identity() ||
    packed_timestep.request_digest() != request.request_digest() ||
    values.length() != packed_timestep.distinct_timestep_count() {
    raise InvalidDistinctTimestepValues
  }
  let owned : Array[Float] = []
  let canonical = StringBuilder()
  canonical <+
    "lunaflux.joint-distinct-timestep-values.v1|model=\{request.identity().content().as_hex()}:\{request.identity().plan().as_hex()}|request=\{request.request_digest().as_hex()}|selection=\{packed_timestep.digest().as_hex()}|values="
  for offset, value in values {
    if value.is_nan() || value.is_inf() || value < 0.0F || value > 1.0F {
      raise InvalidDistinctTimestepValues
    }
    if offset > 0 {
      canonical.write_char(',')
    }
    canonical.write_string(value.reinterpret_as_uint().to_string())
    owned.push(value)
  }
  let digest = try! @spec.PlanDigest::from_sha256(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(@utf8.encode(canonical.to_string())),
    ),
  )
  {
    identity: request.identity(),
    request_digest: request.request_digest(),
    packed_timestep_digest: packed_timestep.digest(),
    digest,
    values: ReadOnlyArray::from_array(owned),
  }
}

///|
pub fn JointDistinctTimestepValues::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn JointDistinctTimestepValues::request_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.request_digest
}

///|
pub fn JointDistinctTimestepValues::packed_timestep_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.packed_timestep_digest
}

///|
pub fn JointDistinctTimestepValues::digest(self : Self) -> @spec.PlanDigest {
  self.digest
}

///|
pub fn JointDistinctTimestepValues::values(self : Self) -> ReadOnlyArray[Float] {
  self.values
}
