///|
/// Producer transport is independent of accumulator/epilogue ownership. These
/// are generic publication plans, not device instruction names. Baseline keeps
/// existing artifacts stable; explicit alternatives are measured offline.
pub(all) enum ProjectionSiblingTransport {
  BaselineSiblingTransport
  SegmentedSiblingTransport
  IndependentSiblingTransport
} derive(Debug, Eq)

///|
pub extend ProjectionSiblingTransport with @debug.Debug::{to_repr}

///|
pub extend ProjectionSiblingTransport with Eq::{equal, not_equal}
