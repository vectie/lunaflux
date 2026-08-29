///|
fn LunaHttp1View::require_workspace(
  self : LunaHttp1View,
) -> LunaHttp1Workspace raise LunaHttp1Error {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_HTTP1_VIEW_LEASED {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Stale)
  }
  self.workspace
}

///|
/// Returns whether this view still owns the workspace's current lease.
pub fn LunaHttp1View::is_live(self : LunaHttp1View) -> Bool {
  self.epoch != 0UL &&
  self.workspace.epoch == self.epoch &&
  self.workspace.phase == LUNA_HTTP1_VIEW_LEASED
}

///|
pub fn LunaHttp1View::route(
  self : LunaHttp1View,
) -> LunaHttp1Route raise LunaHttp1Error {
  match self.require_workspace().route_code {
    1 => LunaHttp1Responses
    2 => LunaHttp1ChatCompletions
    _ => abort("Luna HTTP/1 route invariant")
  }
}

///|
pub fn LunaHttp1View::body_length(
  self : LunaHttp1View,
) -> Int raise LunaHttp1Error {
  self.require_workspace().body_length
}

///|
pub fn LunaHttp1View::body_byte_at(
  self : LunaHttp1View,
  index : Int,
) -> Byte raise LunaHttp1Error {
  let workspace = self.require_workspace()
  if index < 0 || index >= workspace.body_length {
    raise luna_http1_error(LunaHttp1Body, LunaHttp1OutOfRange)
  }
  workspace.storage[workspace.limits.max_head_bytes + index]
}

///|
pub fn LunaHttp1View::release(
  self : LunaHttp1View,
) -> Unit raise LunaHttp1Error {
  let workspace = self.require_workspace()
  workspace.failure_rule = 0
  workspace.failure_issue = 0
  workspace.last_work_units = 0
  workspace.phase = LUNA_HTTP1_IDLE
}
