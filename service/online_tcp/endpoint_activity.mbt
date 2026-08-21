///|
fn LunaOnlineTcpActivity::enter_reactor(
  self : LunaOnlineTcpActivity,
) -> Unit raise LunaOnlineTcpError {
  if self.reactor_active || self.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  self.reactor_active = true
}

///|
fn LunaOnlineTcpActivity::leave_reactor(self : LunaOnlineTcpActivity) -> Unit {
  if !self.reactor_active || self.maintenance_active {
    abort("Luna online TCP reactor activity invariant")
  }
  self.reactor_active = false
}

///|
fn LunaOnlineTcpActivity::enter_maintenance(
  self : LunaOnlineTcpActivity,
) -> Unit raise LunaOnlineTcpError {
  if self.reactor_active || self.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  self.maintenance_active = true
}

///|
fn LunaOnlineTcpActivity::leave_maintenance(
  self : LunaOnlineTcpActivity,
) -> Unit {
  if self.reactor_active || !self.maintenance_active {
    abort("Luna online TCP maintenance activity invariant")
  }
  self.maintenance_active = false
}
