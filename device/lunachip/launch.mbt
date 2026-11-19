///|
/// One exclusive accelerator owner. Transport writes START only after begin;
/// any uncertain write or timeout must call invalidate. No implicit retry.
pub struct LaunchOwner {
  priv mut pending : Bool
  priv mut poisoned : Bool
  priv mut previous : UInt32
  priv mut expected : UInt32
  priv mut count : UInt32
}

///|
pub fn LaunchOwner::new() -> Self {
  { pending: false, poisoned: false, previous: 0U, expected: 0U, count: 0U, }
}

///|
pub fn LaunchOwner::begin(
  self : Self,
  sequence : UInt32,
  count : Int32,
  chain~ : Bool = false,
) -> UInt32 raise AdapterError {
  if self.pending || self.poisoned {
    raise InvalidState
  }
  if count < 1 || count > (if chain { 65535 } else { 2048 }) {
    raise InvalidProgram
  }
  self.previous = sequence
  self.expected = sequence + 1U
  self.count = count.reinterpret_as_uint()
  self.pending = true
  self.expected
}

///|
pub fn LaunchOwner::invalidate(self : Self) -> Unit {
  self.poisoned = true
}

///|
/// Sequence reads bracket status and program count. A changing snapshot or
/// old DONE means wait. A foreign launch, fault or bad completion poisons owner.
pub fn LaunchOwner::observe(
  self : Self,
  sequence_before : UInt32,
  sequence_after : UInt32,
  status : UInt32,
  error : UInt32,
  program_count : UInt32,
) -> Bool raise AdapterError {
  if !self.pending || self.poisoned {
    raise InvalidState
  }
  if sequence_before != sequence_after {
    return false
  }
  if (status & 1U) == 0U || (status & 8U) != 0U || error != 0U {
    self.invalidate()
    raise DeviceFault
  }
  if sequence_after == self.previous {
    return false
  }
  if sequence_after != self.expected ||
    program_count != self.count ||
    (status & 6U) == 6U {
    self.invalidate()
    raise LaunchMismatch
  }
  if (status & 4U) != 0U {
    self.pending = false
    return true
  }
  false
}
