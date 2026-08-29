///|
fn ChildProcess::write_source_conflicts_active_read(
  self : ChildProcess,
  source : FixedArray[Byte],
) -> Bool {
  self.read_transfer_active &&
  (
    physical_equal(source, self.read_transfer_destination) ||
    physical_equal(source, self.read_transfer_staging)
  )
}

///|
fn ChildProcess::read_buffer_conflicts_active_write(
  self : ChildProcess,
  buffer : FixedArray[Byte],
) -> Bool {
  self.write_transfer_active &&
  physical_equal(buffer, self.write_transfer_source)
}

///|
fn InheritedChannel::write_source_conflicts_active_read(
  self : InheritedChannel,
  source : FixedArray[Byte],
) -> Bool {
  self.read_transfer_active &&
  (
    physical_equal(source, self.read_transfer_destination) ||
    physical_equal(source, self.read_transfer_staging)
  )
}

///|
fn InheritedChannel::read_buffer_conflicts_active_write(
  self : InheritedChannel,
  buffer : FixedArray[Byte],
) -> Bool {
  self.write_transfer_active &&
  physical_equal(buffer, self.write_transfer_source)
}
