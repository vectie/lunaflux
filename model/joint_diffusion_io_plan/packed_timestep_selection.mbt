///|
pub fn JointPackedTimestepSelection::new(
  request : JointDiffusionIoPlan,
  evaluation_index~ : Int,
  distinct_timestep_count~ : Int,
  indices~ : ArrayView[Int],
) -> Self raise JointDiffusionIoPlanError {
  let packed_row_count = indices.length()
  if evaluation_index < 0 ||
    evaluation_index >= request.shape().model_evaluations() ||
    distinct_timestep_count <= 0 ||
    distinct_timestep_count > 2147483647 ||
    packed_row_count <= 0 ||
    packed_row_count > 2147483647 {
    raise InvalidPackedTimestepSelection
  }
  let owned : Array[Int] = []
  let canonical = StringBuilder()
  canonical <+
    "lunaflux.joint-packed-timestep-selection.v1|model=\{request.identity().content().as_hex()}:\{request.identity().plan().as_hex()}|request=\{request.request_digest().as_hex()}|evaluation=\{evaluation_index}|rows=\{packed_row_count}|distinct=\{distinct_timestep_count}|indices="
  for offset, index in indices {
    if index < 0 || index >= distinct_timestep_count {
      raise InvalidPackedTimestepSelection
    }
    if offset > 0 {
      canonical.write_char(',')
    }
    canonical.write_string(index.to_string())
    owned.push(index)
  }
  let digest = try! @spec.PlanDigest::from_sha256(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(@utf8.encode(canonical.to_string())),
    ),
  )
  {
    identity: request.identity(),
    request_digest: request.request_digest(),
    digest,
    evaluation_index,
    packed_row_count,
    distinct_timestep_count,
    indices: ReadOnlyArray::from_array(owned),
  }
}

///|
pub fn JointPackedTimestepSelection::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn JointPackedTimestepSelection::request_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.request_digest
}

///|
pub fn JointPackedTimestepSelection::digest(self : Self) -> @spec.PlanDigest {
  self.digest
}

///|
pub fn JointPackedTimestepSelection::evaluation_index(self : Self) -> Int {
  self.evaluation_index
}

///|
pub fn JointPackedTimestepSelection::packed_row_count(self : Self) -> Int {
  self.packed_row_count
}

///|
pub fn JointPackedTimestepSelection::distinct_timestep_count(
  self : Self,
) -> Int {
  self.distinct_timestep_count
}

///|
pub fn JointPackedTimestepSelection::indices(self : Self) -> ReadOnlyArray[Int] {
  self.indices
}
