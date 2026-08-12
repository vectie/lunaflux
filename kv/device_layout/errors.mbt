///|
/// Bounded startup input rejected by the canonical device-KV planner.
pub(all) enum DeviceKvLimitRule {
  TokensPerPage
  TotalPageCount
  ArenaBytes
} derive(Debug, Eq)

///|
/// Semantic model geometry unsupported by this physical layout version.
pub(all) enum DeviceKvGeometryRule {
  DType
  Layout
} derive(Debug, Eq)

///|
/// Checked arithmetic stage that exceeded the signed device-byte envelope.
pub(all) enum DeviceKvSizeStage {
  PageElements
  PageLogicalBytes
  PageStride
  ComponentStride
  LayerStride
  ArenaBytes
  RegionOffset
} derive(Debug, Eq)

///|
/// Dimension used by one rejected scalar region lookup.
pub(all) enum DeviceKvIndexKind {
  Layer
  Page
  Token
  KvHead
  HeadDimension
} derive(Debug, Eq)

///|
/// Payload-safe failures from pure persistent-KV layout planning.
///
/// Errors contain only fixed semantic categories. They never contain model
/// data, device addresses, native diagnostics, or allocation handles.
pub(all) suberror DeviceKvLayoutError {
  InvalidLimit(DeviceKvLimitRule)
  UnsupportedGeometry(DeviceKvGeometryRule)
  SizeOverflow(DeviceKvSizeStage)
  ArenaLimitExceeded
  IndexOutOfBounds(DeviceKvIndexKind)
} derive(Debug, Eq)
