///|
/// One-shot synchronous capability for lending a family-owned host arena.
/// The callback reports success without importing a device/error vocabulary.
pub struct SegmentedHostArenaConsumer {
  priv arena_ordinal : Int
  priv expected_length : Int
  priv consume : (FixedArray[Byte]) -> Bool
  priv mut open : Bool
  priv mut accepted : Bool
}

///|
pub(all) suberror SegmentedHostBorrowError {
  InvalidConsumer
  BorrowClosed(Int)
  ArenaAlreadyProvided(Int)
  ArenaLengthMismatch(Int)
  ArenaNotProvided(Int)
  SourceUnavailable(Int)
  ConsumerRejected(Int)
} derive(Debug, Eq)
