///|
/// Canonical identity of one complete same-host tensor-parallel bootstrap.
pub struct TensorParallelGroupBootstrapDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Canonical identity of the admitted same-host topology.
pub struct TensorParallelTopologyDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Canonical identity of one exact dense-worker startup contract.
pub struct TensorParallelWorkerContractDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Canonical identity of one exact rank-local execution view.
pub struct TensorParallelDevicePlanDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Wire identity copied from one output-only rank-local KV binding proof.
pub struct TensorParallelKvBindingDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Immutable, authority-free rank-specific bootstrap envelope. The opaque
/// rendezvous value grants no device, communicator, context, or root access.
pub struct TensorParallelRankBootstrapEnvelope {
  priv identity : @spec.ModelIdentity
  priv model_generation_value : UInt64
  priv group_generation_value : UInt64
  priv rank : Int
  priv world_size : Int
  priv device_ordinal : Int
  priv collective_runtime_version_code : Int
  priv group_digest : TensorParallelGroupBootstrapDigest
  priv topology_digest : TensorParallelTopologyDigest
  priv worker_digest : TensorParallelWorkerContractDigest
  priv device_plan_digest : TensorParallelDevicePlanDigest
  priv execution_plan_digest_sha256 : String
  priv collective_digest : @device.TensorCollectiveContractDigest
  priv kv_digest : TensorParallelKvBindingDigest
  priv rendezvous_id : @device.TensorCollectiveGroupId
}

///|
/// Immutable startup admission for one complete, rank-ordered local group.
/// It owns no worker, plan buffer, communicator, scheduler state, or backend
/// handle.
pub struct TensorParallelGroupBootstrap {
  priv identity : @spec.ModelIdentity
  priv model_generation_value : UInt64
  priv group_generation_value : UInt64
  priv collective_runtime_version_code : Int
  priv topology_digest : TensorParallelTopologyDigest
  priv digest : TensorParallelGroupBootstrapDigest
  priv template_digest_sha256 : String
  priv envelopes : ReadOnlyArray[TensorParallelRankBootstrapEnvelope]
}

///|
/// Startup-owned fixed storage for one rank envelope frame.
pub struct TensorParallelRankEnvelopeFrameBuffer {
  priv storage : FixedArray[Byte]
  priv mut epoch : UInt64
  priv mut length : Int
}

///|
/// Authenticated immutable view of one frame-buffer epoch.
#valtype
pub struct ValidatedTensorParallelRankEnvelopeFrame(
  TensorParallelRankEnvelopeFrameBuffer,
  UInt64
)

///|
priv struct RankBootstrapClaim {
  identity : @spec.ModelIdentity
  model_generation_value : UInt64
  group_generation_value : UInt64
  rank : Int
  world_size : Int
  device_ordinal : Int
  worker_digest : TensorParallelWorkerContractDigest
  device_plan_digest : TensorParallelDevicePlanDigest
  execution_plan_digest_sha256 : String
  collective_digest : @device.TensorCollectiveContractDigest
  kv_digest : TensorParallelKvBindingDigest
}
