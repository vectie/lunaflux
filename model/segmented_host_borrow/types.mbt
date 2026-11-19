///|
/// One-shot synchronous capability for lending a family-owned host arena.
/// The callback reports success without importing a device/error vocabulary.
pub struct SegmentedHostArenaConsumer {
  priv arena_ordinal : Int32
  priv expected_length : Int32
  priv consume : (FixedArray[Byte]) -> Bool
  priv mut open : Bool
  priv mut accepted : Bool
}

///|
pub(all) suberror SegmentedHostBorrowError {
  InvalidConsumer
  BorrowClosed(Int32)
  ArenaAlreadyProvided(Int32)
  ArenaLengthMismatch(Int32)
  ArenaNotProvided(Int32)
  SourceUnavailable(Int32)
  ConsumerRejected(Int32)
} derive(Debug, Eq)
