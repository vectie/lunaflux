///|
pub fn PhysicalTopologyDeviceObservation::new(
  ordinal~ : Int,
  name~ : StringView,
  capability~ : @device.DeviceCapability,
) -> PhysicalTopologyDeviceObservation raise PhysicalTopologyDiagnosticError {
  if ordinal < 0 || ordinal > 1023 || name.is_empty() || name.length() > 256 {
    raise InvalidObservation
  }
  { ordinal, name: name.to_owned(), capability }
}

///|
pub fn PhysicalPeerObservation::new(
  source_ordinal~ : Int,
  destination_ordinal~ : Int,
  accessible~ : Bool,
) -> PhysicalPeerObservation raise PhysicalTopologyDiagnosticError {
  if source_ordinal < 0 ||
    destination_ordinal < 0 ||
    source_ordinal == destination_ordinal {
    raise InvalidObservation
  }
  { source_ordinal, destination_ordinal, accessible }
}

///|
pub fn PhysicalCollectiveObservation::new(
  availability~ : @device.TensorCollectiveRuntimeAvailability,
  version_code~ : Int,
  admitted_version_code~ : Int,
) -> PhysicalCollectiveObservation raise PhysicalTopologyDiagnosticError {
  match availability {
    CollectiveRuntimeAvailable =>
      if version_code <= 0 ||
        (admitted_version_code != 0 && admitted_version_code != version_code) {
        raise InvalidObservation
      }
    _ =>
      if version_code != 0 || admitted_version_code != 0 {
        raise InvalidObservation
      }
  }
  { availability, version_code, admitted_version_code }
}

///|
pub fn PhysicalTopologyDeviceObservation::ordinal(
  self : PhysicalTopologyDeviceObservation,
) -> Int {
  self.ordinal
}

///|
pub fn PhysicalTopologyDeviceObservation::name(
  self : PhysicalTopologyDeviceObservation,
) -> String {
  self.name
}

///|
pub fn PhysicalTopologyDeviceObservation::capability(
  self : PhysicalTopologyDeviceObservation,
) -> @device.DeviceCapability {
  self.capability
}

///|
pub fn PhysicalPeerObservation::source_ordinal(
  self : PhysicalPeerObservation,
) -> Int {
  self.source_ordinal
}

///|
pub fn PhysicalPeerObservation::destination_ordinal(
  self : PhysicalPeerObservation,
) -> Int {
  self.destination_ordinal
}

///|
pub fn PhysicalPeerObservation::accessible(
  self : PhysicalPeerObservation,
) -> Bool {
  self.accessible
}

///|
pub fn PhysicalCollectiveObservation::availability(
  self : PhysicalCollectiveObservation,
) -> @device.TensorCollectiveRuntimeAvailability {
  self.availability
}

///|
pub fn PhysicalCollectiveObservation::version_code(
  self : PhysicalCollectiveObservation,
) -> Int {
  self.version_code
}

///|
pub fn PhysicalCollectiveObservation::admitted_version_code(
  self : PhysicalCollectiveObservation,
) -> Int {
  self.admitted_version_code
}
