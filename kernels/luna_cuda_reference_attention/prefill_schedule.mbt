///|
/// Explicit terminal alternatives for the pinned paged reference adapter.
/// Same output ownership and numerical permission; different reduction tiles
/// are not claimed bitwise equivalent. Selection is offline, never per token.
pub(all) enum ReferencePrefillSchedule {
  Query64Kv128
  Query64Kv64
} derive(Debug, Eq)

///|
pub fn ReferencePrefillSchedule::key_tile_rows(
  self : ReferencePrefillSchedule,
) -> Int {
  match self {
    Query64Kv128 => 128
    Query64Kv64 => 64
  }
}

///|
/// Q plus independent K/V staging, BF16, head dimension 128. Source lowering
/// verifies this resource contract against the pinned library's static trait.
pub fn ReferencePrefillSchedule::shared_memory_bytes(
  self : ReferencePrefillSchedule,
) -> Int {
  (64 + 2 * self.key_tile_rows()) * 128 * 2
}
