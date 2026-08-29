///|
pub fn PhysicalTopologyDiagnosticEvidence::schema_version(
  _self : PhysicalTopologyDiagnosticEvidence,
) -> Int {
  PHYSICAL_TOPOLOGY_SCHEMA_VERSION
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::status(
  self : PhysicalTopologyDiagnosticEvidence,
) -> PhysicalTopologyDiagnosticStatus {
  if self.rejection is None {
    PhysicalTopologySupported
  } else {
    PhysicalTopologyRejected
  }
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::driver_version(
  self : PhysicalTopologyDiagnosticEvidence,
) -> Int {
  self.driver_version
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::device_count(
  self : PhysicalTopologyDiagnosticEvidence,
) -> Int {
  self.devices.length()
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::device(
  self : PhysicalTopologyDiagnosticEvidence,
  index : Int,
) -> PhysicalTopologyDeviceObservation? {
  self.devices.get(index)
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::peer_count(
  self : PhysicalTopologyDiagnosticEvidence,
) -> Int {
  self.peers.length()
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::peer(
  self : PhysicalTopologyDiagnosticEvidence,
  index : Int,
) -> PhysicalPeerObservation? {
  self.peers.get(index)
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::peer_query_complete(
  self : PhysicalTopologyDiagnosticEvidence,
) -> Bool {
  self.peer_query_complete
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::collective(
  self : PhysicalTopologyDiagnosticEvidence,
) -> PhysicalCollectiveObservation {
  self.collective
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::admitted_topology(
  self : PhysicalTopologyDiagnosticEvidence,
) -> @device_topology.AdmittedLocalTopology? {
  self.admitted_topology
}

///|
pub fn PhysicalTopologyDiagnosticEvidence::rejection(
  self : PhysicalTopologyDiagnosticEvidence,
) -> PhysicalTopologyDiagnosticRejection? {
  self.rejection
}
