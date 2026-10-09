///|
pub(all) enum RemoteStartupRule {
  Count
  Model
  Ordinal
  Generation
  Predecessor
  Limits
  Rendezvous
  Lease
  WorkerSchema
  Destination
  FrameLength
  FrameMismatch
} derive(Debug, Eq)

///|
pub(all) suberror RemoteStartupError {
  Rejected(RemoteStartupRule)
} derive(Debug, Eq)

///|
/// Immutable startup message, not a transport or a device readiness capability.
/// The bytes are independently reproducible from an approved startup bundle.
pub struct RemoteRankStartup {
  priv plan_digest : @two_node_plan.TwoNodePlanDigest
  priv frame : Bytes
  priv worker : @worker_wire.WorkerStartupContract
  priv node : @network_topology.NetworkRank
  priv rank : Int
  priv group_generation : UInt64
  priv group_digest : String
  priv lease_millis : UInt64
  priv rendezvous : @device.TensorCollectiveGroupId
}

///|
pub fn RemoteRankStartup::plan_digest(
  self : RemoteRankStartup,
) -> @two_node_plan.TwoNodePlanDigest {
  self.plan_digest
}

///|
pub struct RemoteGroupStartup {
  priv ranks : ReadOnlyArray[RemoteRankStartup]
  priv digest : String
}

///|
pub fn RemoteGroupStartup::rank(
  self : RemoteGroupStartup,
  rank : Int,
) -> RemoteRankStartup? {
  self.ranks.get(rank)
}

///|
pub fn RemoteGroupStartup::digest_sha256(self : RemoteGroupStartup) -> String {
  self.digest
}

///|
pub fn RemoteRankStartup::rank(self : RemoteRankStartup) -> Int {
  self.rank
}

///|
pub fn RemoteRankStartup::node(
  self : RemoteRankStartup,
) -> @network_topology.NetworkRank {
  self.node
}

///|
pub fn RemoteRankStartup::worker(
  self : RemoteRankStartup,
) -> @worker_wire.WorkerStartupContract {
  self.worker
}

///|
pub fn RemoteRankStartup::group_generation_value(
  self : RemoteRankStartup,
) -> UInt64 {
  self.group_generation
}

///|
pub fn RemoteRankStartup::group_digest_sha256(
  self : RemoteRankStartup,
) -> String {
  self.group_digest
}

///|
pub fn RemoteRankStartup::lease_millis(self : RemoteRankStartup) -> UInt64 {
  self.lease_millis
}

///|
pub fn RemoteRankStartup::rendezvous(
  self : RemoteRankStartup,
) -> @device.TensorCollectiveGroupId {
  self.rendezvous
}

///|
pub extend RemoteStartupRule with Eq::{equal, not_equal}

///|
/// Prepare this generation's value-only watchdog using the local worker clock.
/// The remote resource owner must poll and enforce it independently of
/// collective progress; constructing it does not fence any resources.
pub fn RemoteRankStartup::start_lease(
  self : RemoteRankStartup,
  local_now_millis : UInt64,
) -> @remote_lease.RemoteLease raise RemoteStartupError {
  @remote_lease.RemoteLease::start(
    generation=self.group_generation,
    duration_millis=self.lease_millis,
    local_now_millis~,
  ) catch {
    _ => raise Rejected(Lease)
  }
}

///|
pub extend RemoteStartupRule with @debug.Debug::{to_repr}

///|
pub extend RemoteStartupError with Eq::{equal, not_equal}

///|
pub extend RemoteStartupError with @debug.Debug::{to_repr}
