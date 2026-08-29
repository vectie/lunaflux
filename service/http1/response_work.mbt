///|
pub fn LunaHttp1ResponseWork::progress(
  self : LunaHttp1ResponseWork,
) -> LunaHttp1ResponseProgress raise LunaHttp1Error {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  let budget = workspace.budget.work_units()
  let mut used = 0
  while used < budget && workspace.phase == LUNA_HTTP1_RESPONSE_BUILDING {
    workspace.storage[workspace.cursor] = luna_http1_response_byte(
      workspace.response_kind,
      workspace.cursor,
    )
    workspace.cursor += 1
    used += 1
    if workspace.cursor == workspace.response_length {
      workspace.phase = LUNA_HTTP1_RESPONSE_FINALIZING
    }
  }
  if used < budget && workspace.phase == LUNA_HTTP1_RESPONSE_FINALIZING {
    workspace.phase = LUNA_HTTP1_RESPONSE_READY
    used += 1
  }
  workspace.last_work_units = used
  workspace.total_work_units += used.to_uint64()
  if used == 0 {
    LunaHttp1ResponseIdle
  } else {
    LunaHttp1ResponseAdvanced
  }
}

///|
pub fn LunaHttp1ResponseWork::state(
  self : LunaHttp1ResponseWork,
) -> LunaHttp1ResponseWorkState raise LunaHttp1Error {
  if self.require_workspace().phase == LUNA_HTTP1_RESPONSE_READY {
    LunaHttp1ResponseReady
  } else {
    LunaHttp1ResponseBuilding
  }
}

///|
pub fn LunaHttp1ResponseWork::last_work_units(
  self : LunaHttp1ResponseWork,
) -> Int raise LunaHttp1Error {
  self.require_workspace().last_work_units
}

///|
pub fn LunaHttp1ResponseWork::total_work_units(
  self : LunaHttp1ResponseWork,
) -> UInt64 raise LunaHttp1Error {
  self.require_workspace().total_work_units
}

///|
pub fn LunaHttp1ResponseWork::take_view(
  self : LunaHttp1ResponseWork,
) -> LunaHttp1ResponseView raise LunaHttp1Error {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_HTTP1_RESPONSE_READY {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1NotReady)
  }
  workspace.phase = LUNA_HTTP1_RESPONSE_VIEW_LEASED
  { workspace, epoch: self.epoch }
}

///|
pub fn LunaHttp1ResponseWork::abort(
  self : LunaHttp1ResponseWork,
) -> Unit raise LunaHttp1Error {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  workspace.phase = LUNA_HTTP1_RESPONSE_IDLE
}
