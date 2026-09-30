///|
/// Two physical regions with explicit overlapping or disjoint lifetimes.
/// Reuse requires a producer-reader retirement before consumer publication;
/// lowering owns the device-specific realization of that transition.
pub struct SequentialScratch {
  priv producer_bytes : Int64
  priv consumer_bytes : Int64
  priv overlapping : Bool
} derive(Debug, Eq)

///|
pub fn SequentialScratch::new(
  producer_bytes~ : Int64,
  consumer_bytes~ : Int64,
  overlapping~ : Bool,
) -> SequentialScratch? {
  if producer_bytes <= 0L ||
    consumer_bytes <= 0L ||
    producer_bytes > 9223372036854775807L - consumer_bytes {
    return None
  }
  Some({ producer_bytes, consumer_bytes, overlapping, })
}

///|
pub fn SequentialScratch::reuses_storage(self : SequentialScratch) -> Bool {
  !self.overlapping
}

///|
pub fn SequentialScratch::allocation_bytes(self : SequentialScratch) -> Int64 {
  if self.overlapping {
    self.producer_bytes + self.consumer_bytes
  } else {
    Int64::max(self.producer_bytes, self.consumer_bytes)
  }
}
