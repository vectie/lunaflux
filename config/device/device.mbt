///|
/// Fields accepted by the device configuration constructor.
pub(all) enum DeviceField {
  VisibleDevices
  DeviceId
  MemoryCeilingBytes
} derive(Debug, Eq)

///|
/// Bounded validation categories for device settings.
pub(all) enum DeviceIssue {
  Empty
  TooMany
  BelowMinimum
  AboveMaximum
  Duplicate
} derive(Debug, Eq)

///|
/// Payload-safe device configuration validation failure.
pub(all) suberror DeviceConfigError {
  Invalid(field~ : DeviceField, issue~ : DeviceIssue, index~ : Int)
} derive(Debug, Eq)

///|
/// Immutable assignment and memory ceiling for local accelerators.
pub struct DeviceConfig {
  visible_devices : ReadOnlyArray[Int]
  memory_ceiling_bytes : Int64
} derive(Debug, Eq)

///|
/// Constructs validated explicit device settings.
pub fn new(
  visible_devices : ArrayView[Int],
  memory_ceiling_bytes : Int64,
) -> DeviceConfig raise DeviceConfigError {
  if visible_devices.length() == 0 {
    raise Invalid(field=VisibleDevices, issue=Empty, index=0)
  }
  if visible_devices.length() > 16 {
    raise Invalid(field=VisibleDevices, issue=TooMany, index=16)
  }
  for index, device_id in visible_devices {
    if device_id < 0 {
      raise Invalid(field=DeviceId, issue=BelowMinimum, index~)
    }
    if device_id > 1023 {
      raise Invalid(field=DeviceId, issue=AboveMaximum, index~)
    }
    for prior in 0..<index {
      if visible_devices[prior] == device_id {
        raise Invalid(field=DeviceId, issue=Duplicate, index~)
      }
    }
  }
  if memory_ceiling_bytes < 268435456L {
    raise Invalid(field=MemoryCeilingBytes, issue=BelowMinimum, index=0)
  }
  if memory_ceiling_bytes > 1125899906842624L {
    raise Invalid(field=MemoryCeilingBytes, issue=AboveMaximum, index=0)
  }
  {
    visible_devices: ReadOnlyArray::from_array(visible_devices),
    memory_ceiling_bytes,
  }
}

///|
pub fn DeviceConfig::visible_devices(self : DeviceConfig) -> ReadOnlyArray[Int] {
  self.visible_devices
}

///|
pub fn DeviceConfig::memory_ceiling_bytes(self : DeviceConfig) -> Int64 {
  self.memory_ceiling_bytes
}
