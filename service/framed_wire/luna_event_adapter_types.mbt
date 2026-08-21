///|
/// Synchronous compatibility owner over the authoritative cooperative event
/// workspace. The sole retained View pins the workspace until `release`.
pub struct LunaFramedEventAdapter {
  priv workspace : LunaFramedEventWorkspace
  priv views : Array[LunaFramedEventView]
}
