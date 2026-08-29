///|
type NativeApprovedHandle

///|
/// Opaque GC-managed external capability. Only internal ABI packages may use
/// this type; public packages may retain it solely in private fields.
pub type ApprovedHandle = NativeApprovedHandle

///|
/// Opaque native owner of one authenticated executable descriptor. Only the
/// executable-admission package may construct or close this capability.
type NativeApprovedExecutableHandle

///|
/// Internal capability vocabulary shared by executable admission and the
/// private process ABI. No descriptor or path operation is exposed.
pub type ApprovedExecutableHandle = NativeApprovedExecutableHandle
