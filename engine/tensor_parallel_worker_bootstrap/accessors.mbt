///|
pub fn TensorParallelGroupBootstrapDigest::as_sha256(
  self : TensorParallelGroupBootstrapDigest,
) -> String {
  self.hex
}

///|
pub fn TensorParallelTopologyDigest::as_sha256(
  self : TensorParallelTopologyDigest,
) -> String {
  self.hex
}

///|
pub fn TensorParallelWorkerContractDigest::as_sha256(
  self : TensorParallelWorkerContractDigest,
) -> String {
  self.hex
}

///|
pub fn TensorParallelDevicePlanDigest::as_sha256(
  self : TensorParallelDevicePlanDigest,
) -> String {
  self.hex
}

///|
pub fn TensorParallelKvBindingDigest::as_sha256(
  self : TensorParallelKvBindingDigest,
) -> String {
  self.hex
}

///|
pub fn TensorParallelGroupBootstrap::identity(
  self : TensorParallelGroupBootstrap,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn TensorParallelGroupBootstrap::model_generation_value(
  self : TensorParallelGroupBootstrap,
) -> UInt64 {
  self.model_generation_value
}

///|
pub fn TensorParallelGroupBootstrap::group_generation_value(
  self : TensorParallelGroupBootstrap,
) -> UInt64 {
  self.group_generation_value
}

///|
pub fn TensorParallelGroupBootstrap::world_size(
  self : TensorParallelGroupBootstrap,
) -> Int {
  self.envelopes.length()
}

///|
pub fn TensorParallelGroupBootstrap::topology_digest(
  self : TensorParallelGroupBootstrap,
) -> TensorParallelTopologyDigest {
  self.topology_digest
}

///|
pub fn TensorParallelGroupBootstrap::digest(
  self : TensorParallelGroupBootstrap,
) -> TensorParallelGroupBootstrapDigest {
  self.digest
}

///|
/// Canonical pre-resource parent committed by this live group digest.
pub fn TensorParallelGroupBootstrap::template_digest_sha256(
  self : Self,
) -> String {
  self.template_digest_sha256
}

///|
pub fn TensorParallelGroupBootstrap::rank_envelope(
  self : TensorParallelGroupBootstrap,
  rank : Int,
) -> TensorParallelRankBootstrapEnvelope? {
  self.envelopes.get(rank)
}

///|
pub fn TensorParallelRankBootstrapEnvelope::identity(
  self : TensorParallelRankBootstrapEnvelope,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn TensorParallelRankBootstrapEnvelope::model_generation_value(
  self : TensorParallelRankBootstrapEnvelope,
) -> UInt64 {
  self.model_generation_value
}

///|
pub fn TensorParallelRankBootstrapEnvelope::group_generation_value(
  self : TensorParallelRankBootstrapEnvelope,
) -> UInt64 {
  self.group_generation_value
}

///|
pub fn TensorParallelRankBootstrapEnvelope::rank(
  self : TensorParallelRankBootstrapEnvelope,
) -> Int {
  self.rank
}

///|
pub fn TensorParallelRankBootstrapEnvelope::world_size(
  self : TensorParallelRankBootstrapEnvelope,
) -> Int {
  self.world_size
}

///|
pub fn TensorParallelRankBootstrapEnvelope::device_ordinal(
  self : TensorParallelRankBootstrapEnvelope,
) -> Int {
  self.device_ordinal
}

///|
pub fn TensorParallelRankBootstrapEnvelope::collective_runtime_version_code(
  self : TensorParallelRankBootstrapEnvelope,
) -> Int {
  self.collective_runtime_version_code
}

///|
pub fn TensorParallelGroupBootstrap::collective_runtime_version_code(
  self : TensorParallelGroupBootstrap,
) -> Int {
  self.collective_runtime_version_code
}

///|
pub fn TensorParallelRankBootstrapEnvelope::group_digest(
  self : TensorParallelRankBootstrapEnvelope,
) -> TensorParallelGroupBootstrapDigest {
  self.group_digest
}

///|
pub fn TensorParallelRankBootstrapEnvelope::topology_digest(
  self : TensorParallelRankBootstrapEnvelope,
) -> TensorParallelTopologyDigest {
  self.topology_digest
}

///|
pub fn TensorParallelRankBootstrapEnvelope::worker_digest(
  self : TensorParallelRankBootstrapEnvelope,
) -> TensorParallelWorkerContractDigest {
  self.worker_digest
}

///|
pub fn TensorParallelRankBootstrapEnvelope::device_plan_digest(
  self : TensorParallelRankBootstrapEnvelope,
) -> TensorParallelDevicePlanDigest {
  self.device_plan_digest
}

///|
/// Returns the canonical wire form of the execution-plan-owned digest.
///
/// The execution-plan package remains the sole owner of the typed digest. Its
/// versioned hash transitively binds the exact launch and artifact identities.
pub fn TensorParallelRankBootstrapEnvelope::execution_plan_digest_sha256(
  self : TensorParallelRankBootstrapEnvelope,
) -> String {
  self.execution_plan_digest_sha256
}

///|
pub fn TensorParallelRankBootstrapEnvelope::collective_digest(
  self : TensorParallelRankBootstrapEnvelope,
) -> @device.TensorCollectiveContractDigest {
  self.collective_digest
}

///|
pub fn TensorParallelRankBootstrapEnvelope::kv_digest(
  self : TensorParallelRankBootstrapEnvelope,
) -> TensorParallelKvBindingDigest {
  self.kv_digest
}

///|
/// Returns only the existing opaque rendezvous identity needed by the private
/// device owner. No vendor handle or backend payload becomes visible.
pub fn TensorParallelRankBootstrapEnvelope::rendezvous_id(
  self : TensorParallelRankBootstrapEnvelope,
) -> @device.TensorCollectiveGroupId {
  self.rendezvous_id
}
