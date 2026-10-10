///|
/// Coordinate contract shared by compressed-row selection and its consumer.
/// Separate compressed-cache storage uses direct row IDs. Concatenated/joined
/// storage can request a caller-owned per-row offset. This is a pure AOT choice,
/// not a runtime branch or a model-family convention.
pub(all) enum CompressedRowCoordinates {
  CacheRelative
  OffsetRelative
} derive(Debug, Eq)

///|
pub extend CompressedRowCoordinates with @debug.Debug::{to_repr}

///|
pub extend CompressedRowCoordinates with Eq::{equal, not_equal}
