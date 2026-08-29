///|
fn LunaHttp1Workspace::progress_one(self : LunaHttp1Workspace) -> Unit {
  match self.phase {
    LUNA_HTTP1_SCANNING_LINE => self.progress_scan_line_one()
    LUNA_HTTP1_VALIDATING_REQUEST => self.progress_request_one()
    LUNA_HTTP1_VALIDATING_HEADER_NAME => self.progress_header_name_one()
    LUNA_HTTP1_VALIDATING_HEADER_VALUE => self.progress_header_value_one()
    LUNA_HTTP1_VALIDATING_REQUIRED => self.progress_required_one()
    LUNA_HTTP1_AUTH_BEGIN => self.progress_auth_begin_one()
    LUNA_HTTP1_AUTH_WRITE => self.progress_auth_write_one()
    LUNA_HTTP1_AUTH_FINISH => self.progress_auth_finish_one()
    LUNA_HTTP1_AUTH_PROGRESS => self.progress_auth_work_one()
    LUNA_HTTP1_AUTH_DECIDE => self.progress_auth_decide_one()
    LUNA_HTTP1_FINALIZING => self.phase = LUNA_HTTP1_READY
    _ => ()
  }
}

///|
/// Advances no more than one configured quantum of validation or auth work.
pub fn LunaHttp1Work::progress(
  self : LunaHttp1Work,
) -> LunaHttp1Progress raise LunaHttp1Error {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  if workspace.phase == LUNA_HTTP1_FAILED {
    return LunaHttp1Idle
  }
  let mut used = 0
  let budget = workspace.budget.work_units()
  while used < budget &&
        workspace.phase != LUNA_HTTP1_RECEIVING_HEAD &&
        workspace.phase != LUNA_HTTP1_RECEIVING_BODY &&
        workspace.phase != LUNA_HTTP1_READY &&
        workspace.phase != LUNA_HTTP1_FAILED {
    workspace.progress_one()
    used += 1
  }
  workspace.charge(used)
  if used == 0 {
    LunaHttp1Idle
  } else {
    LunaHttp1Advanced
  }
}
