///|
const MIN_LOCAL_RANKS : Int = 2

///|
const MAX_LOCAL_RANKS : Int = 16

///|
const MAX_DEVICE_NAME_UNITS : Int = 256

///|
/// One explicit rank identity. Array position is the stable rank; the ordinal
/// remains the process-visible device identity selected by deployment.
pub struct LocalTopologyRank {
  priv ordinal : Int
  priv name : String
  priv total_memory_bytes : Int64
  priv target : @catalog.DeviceTarget
} derive(Debug, Eq)

///|
/// One canonical undirected link required by the initial local full mesh.
pub struct LocalTopologyLink {
  priv lower_rank : Int
  priv upper_rank : Int
} derive(Debug, Eq)

///|
/// Explicit initial Phase-7 same-host topology declaration.
pub struct LocalTopologyDeclaration {
  priv ranks : ReadOnlyArray[LocalTopologyRank]
  priv links : ReadOnlyArray[LocalTopologyLink]
} derive(Debug, Eq)

///|
/// One directed peer-access observation between process-visible ordinals.
pub struct LocalPeerAccess {
  priv source_ordinal : Int
  priv destination_ordinal : Int
} derive(Debug, Eq)

///|
/// One bounded authority-free device observation for host-testable projection.
pub struct LocalProbeDevice {
  priv ordinal : Int
  priv name : String
  priv total_memory_bytes : Int64
  priv target : @catalog.DeviceTarget
} derive(Debug, Eq)

///|
/// Authority-free projection of a host device probe plus directed peer access.
pub struct LocalTopologyProbeProjection {
  priv driver_version : Int
  priv devices : ReadOnlyArray[LocalProbeDevice]
  priv peer_access : ReadOnlyArray[LocalPeerAccess]
} derive(Debug, Eq)

///|
/// Immutable startup capability. It owns no context, communicator, or worker.
pub struct AdmittedLocalTopology {
  priv driver_version : Int
  priv ranks : ReadOnlyArray[LocalTopologyRank]
  priv target : @catalog.DeviceTarget
  priv link_count : Int
} derive(Debug, Eq)
