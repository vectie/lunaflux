///|
/// Construct a bounded one-shot consumer. Family host owners receive this
/// capability but never receive or import device state.
pub fn SegmentedHostArenaConsumer::new(
  arena_ordinal~ : Int,
  expected_length~ : Int,
  consume~ : (FixedArray[Byte]) -> Bool,
) -> Self raise SegmentedHostBorrowError {
  if arena_ordinal < 0 || expected_length <= 0 {
    raise InvalidConsumer
  }
  { arena_ordinal, expected_length, consume, open: true, accepted: false, }
}

///|
/// Supply exactly one arena while this synchronous capability is live.
pub fn SegmentedHostArenaConsumer::accept(
  self : Self,
  arena : FixedArray[Byte],
) -> Unit raise SegmentedHostBorrowError {
  if !self.open {
    raise BorrowClosed(self.arena_ordinal)
  }
  if self.accepted {
    raise ArenaAlreadyProvided(self.arena_ordinal)
  }
  if arena.length() != self.expected_length {
    raise ArenaLengthMismatch(self.arena_ordinal)
  }
  if !(self.consume)(arena) {
    raise ConsumerRejected(self.arena_ordinal)
  }
  self.accepted = true
}

///|
/// Revoke the capability immediately after the provider callback returns.
pub fn SegmentedHostArenaConsumer::revoke(self : Self) -> Unit {
  self.open = false
}

///|
pub fn SegmentedHostArenaConsumer::was_accepted(self : Self) -> Bool {
  self.accepted
}
