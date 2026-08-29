///|
fn InheritedChannel::wait_inherited_direction(
  self : InheritedChannel,
  write_mode : Bool,
  maximum_wait_millis : Int,
) -> Bool raise ProcessError {
  let operation = if write_mode { Write } else { Read }
  self.require_usable(operation)
  if maximum_wait_millis <= 0 || maximum_wait_millis > 600000 {
    raise InvalidRange(operation)
  }
  let status = raw_inherited_wait(write_mode, maximum_wait_millis)
  match status {
    0 => true
    2 => false
    _ => {
      self.poison_transfers()
      raise_status(status, operation)
      abort("unreachable inherited wait status")
    }
  }
}

///|
fn InheritedChannel::bounded_pending_wait(
  self : InheritedChannel,
  write_mode : Bool,
  maximum_wait_millis : Int,
) -> Bool raise ProcessError {
  let operation = if write_mode { Write } else { Read }
  let active = if write_mode {
    self.write_transfer_active
  } else {
    self.read_transfer_active
  }
  let deadline = if write_mode {
    self.write_transfer_deadline_millis
  } else {
    self.read_transfer_deadline_millis
  }
  if !active {
    raise InvalidState(operation)
  }
  self.require_transfer_unexpired(deadline, operation)
  let now = raw_monotonic_now()
  if now < 0L {
    self.poison_transfers()
    raise OperationFailed(operation)
  }
  let remaining = deadline - now
  let selected = if remaining < maximum_wait_millis.to_int64() {
    remaining.to_int()
  } else {
    maximum_wait_millis
  }
  let ready = self.wait_inherited_direction(write_mode, selected)
  if !ready {
    self.require_transfer_unexpired(deadline, operation)
  }
  ready
}

///|
/// Waits for an active inherited read without extending its absolute deadline.
/// A short caller bound timing out is ordinary pending progress.
pub fn InheritedChannel::wait_pending_readable(
  self : InheritedChannel,
  maximum_wait_millis~ : Int,
) -> Bool raise ProcessError {
  self.bounded_pending_wait(false, maximum_wait_millis)
}

///|
/// Waits for an active inherited write without extending its absolute deadline.
/// A short caller bound timing out is ordinary pending progress.
pub fn InheritedChannel::wait_pending_writable(
  self : InheritedChannel,
  maximum_wait_millis~ : Int,
) -> Bool raise ProcessError {
  self.bounded_pending_wait(true, maximum_wait_millis)
}

///|
/// Performs a bounded readability wait without consuming bytes. This is used
/// only to pace nonblocking device progress before a command read is active.
pub fn InheritedChannel::wait_readable(
  self : InheritedChannel,
  maximum_wait_millis~ : Int,
) -> Bool raise ProcessError {
  self.wait_inherited_direction(false, maximum_wait_millis)
}
