///|
/// Narrow authoritative live-server health projection.
#valtype
pub(all) enum LunaOnlineTcpServerHealth {
  LunaOnlineTcpServerHealthy
  LunaOnlineTcpServerFailed
} derive(Debug, Eq)
