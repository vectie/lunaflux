///|
fn LunaOnlineTcpMaintenanceHandoff::arm(
  self : LunaOnlineTcpMaintenanceHandoff,
) -> Unit {
  self.pending = true
}

///|
fn LunaOnlineTcpMaintenanceHandoff::is_pending(
  self : LunaOnlineTcpMaintenanceHandoff,
) -> Bool {
  self.pending
}

///|
fn LunaOnlineTcpMaintenanceHandoff::require_pending(
  self : LunaOnlineTcpMaintenanceHandoff,
) -> Unit raise LunaOnlineTcpError {
  if !self.pending {
    raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
  }
}

///|
fn LunaOnlineTcpMaintenanceHandoff::clear(
  self : LunaOnlineTcpMaintenanceHandoff,
) -> Unit {
  self.pending = false
}
