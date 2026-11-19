///|
const PHYSICAL_TOPOLOGY_SCHEMA_VERSION : Int32 = 1

///|
const MINIMUM_COLLECTIVE_RUNTIME_VERSION : Int32 = 21403

///|
const MAXIMUM_COLLECTIVE_RUNTIME_VERSION : Int32 = 29999

///|
pub(all) enum PhysicalTopologyDiagnosticStage {
  DeviceProbe
  PeerProbe
  TopologyProjection
  TopologyDeclaration
  TopologyAdmission
  CollectiveRuntime
} derive(Debug, Eq)

///|
pub(all) enum PhysicalTopologyDiagnosticStatus {
  PhysicalTopologySupported
  PhysicalTopologyRejected
} derive(Debug, Eq)

///|
pub(all) enum PhysicalTopologyDiagnosticRejection {
  DeviceUnavailable(@device.UnavailableReason)
  PeerQueryFailed(source_rank~ : Int32, destination_rank~ : Int32)
  TopologyRejected(
    stage~ : PhysicalTopologyDiagnosticStage,
    area~ : @device_topology.LocalTopologyArea,
    rule~ : @device_topology.LocalTopologyRule,
    first~ : Int32,
    second~ : Int32
  )
  CollectiveUnavailable(@device.TensorCollectiveRuntimeAvailability)
  CollectiveVersionRejected
} derive(Debug, Eq)

///|
pub(all) suberror PhysicalTopologyDiagnosticError {
  InvalidObservation
} derive(Debug, Eq)

///|
pub struct PhysicalTopologyDeviceObservation {
  priv ordinal : Int32
  priv name : String
  priv capability : @device.DeviceCapability
} derive(Debug, Eq)

///|
pub struct PhysicalPeerObservation {
  priv source_ordinal : Int32
  priv destination_ordinal : Int32
  priv accessible : Bool
} derive(Debug, Eq)

///|
pub struct PhysicalCollectiveObservation {
  priv availability : @device.TensorCollectiveRuntimeAvailability
  priv version_code : Int32
  priv admitted_version_code : Int32
} derive(Debug, Eq)

///|
/// Immutable, authority-free Phase-7 diagnostic evidence. The exact topology
/// admission is retained whenever that component passes, including when the
/// collective runtime independently rejects. No runtime owner is retained in
/// either outcome.
pub struct PhysicalTopologyDiagnosticEvidence {
  priv driver_version : Int32
  priv devices : ReadOnlyArray[PhysicalTopologyDeviceObservation]
  priv peers : ReadOnlyArray[PhysicalPeerObservation]
  priv peer_query_complete : Bool
  priv collective : PhysicalCollectiveObservation
  priv admitted_topology : @device_topology.AdmittedLocalTopology?
  priv rejection : PhysicalTopologyDiagnosticRejection?
}
