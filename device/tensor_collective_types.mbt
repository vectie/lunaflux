///|
/// Model-neutral collectives supported by the initial private device owner.
pub(all) enum TensorCollectiveKind {
  Bf16SumAllReduce
  Bf16AllGather
} derive(Debug, Eq)

///|
/// Canonical identity of one immutable rank-local collective contract.
pub struct TensorCollectiveContractDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// One immutable scalar site copied into device-owned startup storage.
pub struct TensorCollectiveSite {
  priv sequence : Int
  priv operation_id : Int
  priv kind : TensorCollectiveKind
  priv full_width : Int
  priv local_width : Int
} derive(Debug, Eq)

///|
/// Model-neutral, immutable per-rank collective contract. Parallel fixed
/// arrays avoid model-family objects and request-path allocation.
pub struct TensorCollectiveContract {
  priv digest : TensorCollectiveContractDigest
  priv rank : Int
  priv world_size : Int
  priv max_query_tokens : Int
  priv sequences : FixedArray[Int]
  priv operation_ids : FixedArray[Int]
  priv kinds : FixedArray[Int]
  priv full_widths : FixedArray[Int]
  priv local_widths : FixedArray[Int]
}

///|
/// Bounded process-lifetime collective runtime inventory. This mirrors no
/// backend result code, path, symbol, or handle.
pub(all) enum TensorCollectiveRuntimeAvailability {
  CollectiveRuntimeAvailable
  CollectiveRuntimeUnsupportedPlatform
  CollectiveRuntimeLibraryMissing
  CollectiveRuntimeAbiIncomplete
  CollectiveRuntimeVersionUnavailable
} derive(Debug, Eq)

///|
/// Authority-free collective runtime observation for startup diagnostics.
pub struct TensorCollectiveRuntimeInventory {
  priv availability : TensorCollectiveRuntimeAvailability
  priv version_code : Int
} derive(Debug, Eq)

///|
/// Root-free admission of the immutable process-lifetime NCCL dispatch.
pub struct TensorCollectiveRuntime {
  priv raw : @nccl.NcclRuntimeAdmission
}

///|
/// Opaque rendezvous identity. It carries no rank, model, or root authority.
pub struct TensorCollectiveGroupId {
  priv raw : @nccl.NcclCommunicatorId
}

///|
enum TensorCollectiveOwnerState {
  CollectiveOwnerStarting
  CollectiveOwnerLive
  CollectiveOwnerInFlight
  CollectiveOwnerFailed
  CollectiveOwnerClosed
} derive(Debug, Eq)

///|
/// Thread-confined owner of one rank communicator and immutable site table.
pub struct TensorCollectiveOwner {
  priv context : Context
  priv contract : TensorCollectiveContract
  priv communicator : @nccl.NcclCommunicator
  priv generation : UInt64
  priv mut state : TensorCollectiveOwnerState
  priv mut plan_sequence : UInt64
  priv mut next_global_sequence : UInt64
  priv mut next_site_index : Int
  priv mut pending_plan_sequence : UInt64
  priv mut pending_global_sequence : UInt64
  priv mut pending_site_index : Int
}
