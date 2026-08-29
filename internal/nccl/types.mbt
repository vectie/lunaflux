///|
/// Bounded dynamic NCCL loader state. Vendor result codes, library paths, and
/// vendor strings never enter this representation.
pub(all) enum NcclAvailability {
  Available
  UnsupportedPlatform
  LibraryMissing
  AbiIncomplete
  VersionUnavailable
} derive(Debug, Eq)

///|
pub struct NcclRuntimeInventory {
  priv availability : NcclAvailability
  priv version_code : Int
} derive(Debug, Eq)

///|
pub fn NcclRuntimeInventory::availability(
  self : NcclRuntimeInventory,
) -> NcclAvailability {
  self.availability
}

///|
pub fn NcclRuntimeInventory::version_code(self : NcclRuntimeInventory) -> Int {
  self.version_code
}

///|
/// Root-free proof that one already-loaded runtime met caller-owned bounds.
pub struct NcclRuntimeAdmission {
  priv version_code : Int
}

///|
pub fn NcclRuntimeAdmission::version_code(self : NcclRuntimeAdmission) -> Int {
  self.version_code
}

///|
/// Opaque fixed-size rendezvous identity. It carries no rank or root authority.
pub struct NcclCommunicatorId {
  priv bytes : FixedArray[Byte]
}

///|
pub(all) enum NcclCollectiveOperation {
  AllReduce
  AllGather
  ReduceScatter
} derive(Debug, Eq)

///|
pub(all) enum NcclOperation {
  RuntimeAdmission
  CommunicatorIdCreate
  CommunicatorCreate
  CollectiveAdmission
  CollectiveExecution
  CommunicatorInvalidate
  CommunicatorClose
  CommunicatorAbort
} derive(Debug, Eq)

///|
/// Payload-safe private-native failure. No vendor code, text, path, or handle
/// can cross this boundary.
pub(all) suberror NcclError {
  Unavailable(NcclOperation)
  VersionMismatch(NcclOperation)
  InvalidArgument(NcclOperation)
  Closed(NcclOperation)
  Busy(NcclOperation)
  RuntimeFailure(NcclOperation)
  GenerationMismatch(NcclOperation)
  SequenceMismatch(NcclOperation)
  Failed(NcclOperation)
} derive(Debug, Eq)

///|
/// One explicitly owned native communicator for one admitted rank.
type NcclCommunicator
