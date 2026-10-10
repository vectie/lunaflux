///|
pub(all) enum TensorParallelRankConfigureRule {
  Limit
  Magic
  Version
  Length
  Checksum
  Reserved
  Digest
  Startup
  BootstrapSource
  Topology
  RankEnvelope
  ModelIdentity
  ModelGeneration
  GroupGeneration
  Rank
  WorldSize
  DeviceOrdinal
  Predecessor
  Binding
  Destination
  FrameEpoch
} derive(Debug, Eq)

///|
pub(all) suberror TensorParallelRankConfigureError {
  Invalid(TensorParallelRankConfigureRule)
  StaleFrame
} derive(Debug, Eq)

///|
/// Already-admitted worker contract identity carried in the legacy digest
/// slot. This is not a checksum of the Configure frame.
pub struct TensorParallelRankConfigureDigest(String) derive(Debug, Eq)

///|
/// Immutable, authority-free, bounded rank Configure contract. Every nested
/// owner is decoded from and structurally matches `storage`. Legacy FNV words
/// are ignored; the private startup channel and locked binding own admission.
pub struct TensorParallelRankConfigureContract {
  priv storage : FixedArray[Byte]
  priv length : Int
  priv digest : TensorParallelRankConfigureDigest
  priv binding : @rank_group_wire.RankGroupWireBinding
  priv startup : @worker_wire.WorkerStartupContract
  priv source : @worker_wire.EncodedBootstrapSource
  priv topology : @device_topology.LocalTopologyDeclaration
  priv envelope : @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope
}

///|
/// Rank-bound receive storage. One buffer can validate successive frames, but
/// a validated view authenticates exactly one nonzero epoch.
pub struct TensorParallelRankConfigureFrameBuffer {
  priv binding : @rank_group_wire.RankGroupWireBinding
  priv storage : FixedArray[Byte]
  priv mut epoch : UInt64
  priv mut length : Int
}

///|
#valtype
pub struct ValidatedTensorParallelRankConfigureFrame(
  TensorParallelRankConfigureFrameBuffer,
  UInt64
)

///|
pub fn TensorParallelRankConfigureDigest::as_sha256(
  self : TensorParallelRankConfigureDigest,
) -> String {
  self.0
}

///|
pub fn TensorParallelRankConfigureContract::length(
  self : TensorParallelRankConfigureContract,
) -> Int {
  self.length
}

///|
pub fn TensorParallelRankConfigureContract::digest(
  self : TensorParallelRankConfigureContract,
) -> TensorParallelRankConfigureDigest {
  self.digest
}

///|
pub fn TensorParallelRankConfigureContract::binding(
  self : TensorParallelRankConfigureContract,
) -> @rank_group_wire.RankGroupWireBinding {
  self.binding
}

///|
pub fn TensorParallelRankConfigureContract::startup(
  self : TensorParallelRankConfigureContract,
) -> @worker_wire.WorkerStartupContract {
  self.startup
}

///|
pub fn TensorParallelRankConfigureContract::bootstrap_source(
  self : TensorParallelRankConfigureContract,
) -> @worker_wire.EncodedBootstrapSource {
  self.source
}

///|
pub fn TensorParallelRankConfigureContract::topology(
  self : TensorParallelRankConfigureContract,
) -> @device_topology.LocalTopologyDeclaration {
  self.topology
}

///|
pub fn TensorParallelRankConfigureContract::rank_envelope(
  self : TensorParallelRankConfigureContract,
) -> @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope {
  self.envelope
}
