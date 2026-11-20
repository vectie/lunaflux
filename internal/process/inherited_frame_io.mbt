///|
fn InheritedChannel::require_usable(
  self : InheritedChannel,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  if self.failed {
    raise InvalidState(operation)
  }
}

///|
fn InheritedChannel::write_exact(
  self : InheritedChannel,
  source : FixedArray[Byte],
  offset : Int32,
  byte_count : Int32,
) -> Unit raise ProcessError {
  self.require_usable(Write)
  if self.write_transfer_active {
    raise InvalidState(Write)
  }
  if self.write_source_conflicts_active_read(source) {
    raise InvalidRange(Write)
  }
  let result = Ok(
    raise_status(
      raw_inherited_write_exact(
        source,
        offset,
        byte_count,
        self.limits.io_timeout_millis,
      ),
      Write,
    ),
  ) catch {
    error => Err(error)
  }
  match result {
    Ok(_) => ()
    Err(error) => {
      self.poison_transfers()
      raise error
    }
  }
}

///|
fn InheritedChannel::read_exact(
  self : InheritedChannel,
  destination : FixedArray[Byte],
  offset : Int32,
  byte_count : Int32,
) -> Unit raise ProcessError {
  self.require_usable(Read)
  if self.read_transfer_active {
    raise InvalidState(Read)
  }
  let result = Ok(
    raise_status(
      raw_inherited_read_exact(
        destination,
        offset,
        byte_count,
        self.limits.io_timeout_millis,
      ),
      Read,
    ),
  ) catch {
    error => Err(error)
  }
  match result {
    Ok(_) => ()
    Err(error) => {
      self.poison_transfers()
      raise error
    }
  }
}

///|
/// Writes one canonical length-delimited worker frame.
pub fn InheritedChannel::write_frame(
  self : InheritedChannel,
  source : FixedArray[Byte],
  offset~ : Int32,
  byte_count~ : Int32,
) -> Unit raise ProcessError {
  self.require_usable(Write)
  if self.write_transfer_active {
    raise InvalidState(Write)
  }
  if self.write_source_conflicts_active_read(source) {
    raise InvalidRange(Write)
  }
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    offset < 0 ||
    offset > source.length() - byte_count {
    raise InvalidRange(Write)
  }
  write_u32_prefix(self.write_prefix, byte_count)
  self.write_exact(self.write_prefix, 0, 4)
  self.write_exact(source, offset, byte_count)
}

///|
/// Waits for a clean parent half-close without consuming framed payload. A
/// single byte, including a partial next-frame prefix, fails and permanently
/// poisons the channel; timeout and native errors fail likewise.
pub fn InheritedChannel::expect_clean_eof(
  self : InheritedChannel,
) -> Unit raise ProcessError {
  self.require_usable(Read)
  if self.read_transfer_active {
    raise InvalidState(Read)
  }
  let result = Ok(
    raise_status(
      raw_inherited_expect_clean_eof(self.limits.io_timeout_millis),
      Read,
    ),
  ) catch {
    error => Err(error)
  }
  match result {
    Ok(_) => ()
    Err(error) => {
      self.poison_transfers()
      raise error
    }
  }
}

///|
/// Reads one canonical length-delimited worker frame.
pub fn InheritedChannel::read_frame(
  self : InheritedChannel,
  destination : FixedArray[Byte],
) -> Int32 raise ProcessError {
  self.require_usable(Read)
  if self.read_transfer_active {
    raise InvalidState(Read)
  }
  if self.read_buffer_conflicts_active_write(destination) {
    raise InvalidRange(Read)
  }
  self.read_exact(self.read_prefix, 0, 4)
  let byte_count = read_u32_prefix(self.read_prefix)
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    byte_count > destination.length() {
    self.poison_transfers()
    raise InvalidRange(Read)
  }
  self.read_exact(destination, 0, byte_count)
  byte_count
}

///|
/// Reads one canonical length-delimited frame or returns zero only when the
/// parent cleanly closes before any byte of the next prefix. EOF after a
/// partial prefix or payload is a terminal channel error. Positive results are
/// exact frame lengths and therefore never overlap the EOF sentinel.
pub fn InheritedChannel::read_frame_or_eof(
  self : InheritedChannel,
  destination : FixedArray[Byte],
) -> Int32 raise ProcessError {
  self.require_usable(Read)
  if self.read_transfer_active {
    raise InvalidState(Read)
  }
  if self.read_buffer_conflicts_active_write(destination) {
    raise InvalidRange(Read)
  }
  let status = raw_inherited_read_prefix_or_eof(
    self.read_prefix,
    self.limits.io_timeout_millis,
  )
  if status == 3 {
    return 0
  }
  let result = Ok(raise_status(status, Read)) catch { error => Err(error) }
  match result {
    Ok(_) => ()
    Err(error) => {
      self.poison_transfers()
      raise error
    }
  }
  let byte_count = read_u32_prefix(self.read_prefix)
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    byte_count > destination.length() {
    self.poison_transfers()
    raise InvalidRange(Read)
  }
  self.read_exact(destination, 0, byte_count)
  byte_count
}
