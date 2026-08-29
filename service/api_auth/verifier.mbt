///|
pub fn LunaApiAuthVerifierWorkspace::new(
  policy : LunaApiAuthPolicy,
  budget : LunaApiAuthStepBudget,
) -> LunaApiAuthVerifierWorkspace raise LunaApiAuthError {
  if !policy.is_authenticated() {
    raise luna_api_auth_error(LunaApiAuthPolicy, LunaApiAuthInvalid)
  }
  {
    policy,
    budget,
    epoch: 0UL,
    phase: LUNA_API_AUTH_IDLE,
    required_input_bytes: 0,
    cursor: 0,
    difference: 0,
    failure_rule: 0,
    failure_issue: 0,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
/// Invalidates every outstanding write/work epoch and wipes the shared policy.
/// Repeated close is harmless; no later verification can restart this owner.
pub fn LunaApiAuthVerifierWorkspace::close(
  self : LunaApiAuthVerifierWorkspace,
) -> Unit {
  if self.phase == LUNA_API_AUTH_CLOSED {
    return
  }
  self.policy.close()
  self.reset_operation()
  self.phase = LUNA_API_AUTH_CLOSED
}

///|
fn LunaApiAuthVerifierWorkspace::reset_operation(
  self : LunaApiAuthVerifierWorkspace,
) -> Unit {
  self.required_input_bytes = 0
  self.cursor = 0
  self.difference = 0
  self.failure_rule = 0
  self.failure_issue = 0
  self.last_work_units = 0
  self.total_work_units = 0UL
}

///|
pub fn LunaApiAuthVerifierWorkspace::begin_verify(
  self : LunaApiAuthVerifierWorkspace,
  declared_length : Int,
) -> LunaApiAuthVerificationWrite raise LunaApiAuthError {
  if self.phase != LUNA_API_AUTH_IDLE {
    raise luna_api_auth_error(LunaApiAuthLifecycle, LunaApiAuthBusy)
  }
  if !self.policy.is_authenticated() {
    raise luna_api_auth_error(LunaApiAuthPolicy, LunaApiAuthInvalid)
  }
  if declared_length < 0 {
    raise luna_api_auth_error(LunaApiAuthCredential, LunaApiAuthOutOfRange)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise luna_api_auth_error(LunaApiAuthEpoch, LunaApiAuthExhausted)
  }
  self.reset_operation()
  self.epoch += 1UL
  self.required_input_bytes = Int::min(
    declared_length,
    self.policy.expected_length,
  )
  self.difference = if declared_length == self.policy.expected_length {
    0
  } else {
    1
  }
  self.phase = LUNA_API_AUTH_WRITING
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaApiAuthVerificationWrite::require_workspace(
  self : LunaApiAuthVerificationWrite,
) -> LunaApiAuthVerifierWorkspace raise LunaApiAuthError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_API_AUTH_WRITING {
    raise luna_api_auth_error(LunaApiAuthEpoch, LunaApiAuthStale)
  }
  self.workspace
}

///|
#inline(never)
pub fn LunaApiAuthVerificationWrite::push_byte(
  self : LunaApiAuthVerificationWrite,
  byte : Byte,
) -> LunaApiAuthWriteProgress raise LunaApiAuthError {
  let workspace = self.require_workspace()
  if !workspace.policy.is_authenticated() {
    raise luna_api_auth_error(LunaApiAuthPolicy, LunaApiAuthInvalid)
  }
  if workspace.cursor >= workspace.required_input_bytes {
    raise luna_api_auth_error(LunaApiAuthLifecycle, LunaApiAuthNotReady)
  }
  workspace.difference = workspace.policy.compare_byte(
    workspace.cursor,
    byte,
    workspace.difference,
  )
  workspace.cursor += 1
  workspace.total_work_units += 1UL
  {
    state: if workspace.cursor == workspace.required_input_bytes {
      1
    } else {
      0
    },
    work_units: 1,
  }
}

///|
pub fn LunaApiAuthVerificationWrite::finish(
  self : LunaApiAuthVerificationWrite,
) -> LunaApiAuthVerificationWork raise LunaApiAuthError {
  let workspace = self.require_workspace()
  if workspace.cursor != workspace.required_input_bytes {
    raise luna_api_auth_error(LunaApiAuthLifecycle, LunaApiAuthNotReady)
  }
  workspace.phase = LUNA_API_AUTH_PADDING
  { workspace, epoch: self.epoch }
}

///|
pub fn LunaApiAuthVerificationWrite::abort(
  self : LunaApiAuthVerificationWrite,
) -> Unit raise LunaApiAuthError {
  let workspace = self.require_workspace()
  workspace.reset_operation()
  workspace.phase = LUNA_API_AUTH_IDLE
}

///|
fn LunaApiAuthVerificationWork::require_workspace(
  self : LunaApiAuthVerificationWork,
) -> LunaApiAuthVerifierWorkspace raise LunaApiAuthError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase == LUNA_API_AUTH_IDLE ||
    self.workspace.phase == LUNA_API_AUTH_WRITING {
    raise luna_api_auth_error(LunaApiAuthEpoch, LunaApiAuthStale)
  }
  self.workspace
}

///|
fn LunaApiAuthVerifierWorkspace::fail(
  self : LunaApiAuthVerifierWorkspace,
  error : LunaApiAuthError,
) -> Unit {
  match error {
    LunaApiAuthFailed(rule~, issue~) => {
      self.failure_rule = match rule {
        LunaApiAuthPolicy => 1
        LunaApiAuthCredential => 2
        LunaApiAuthSource => 3
        LunaApiAuthWorkUnits => 4
        LunaApiAuthLifecycle => 5
        LunaApiAuthEpoch => 6
      }
      self.failure_issue = match issue {
        LunaApiAuthInvalid => 1
        LunaApiAuthEmpty => 2
        LunaApiAuthTooLong => 3
        LunaApiAuthOutOfRange => 4
        LunaApiAuthBusy => 5
        LunaApiAuthNotReady => 6
        LunaApiAuthStale => 7
        LunaApiAuthExhausted => 8
      }
    }
  }
  self.phase = LUNA_API_AUTH_FAILED
}

///|
fn LunaApiAuthVerifierWorkspace::failure_value(
  self : LunaApiAuthVerifierWorkspace,
) -> LunaApiAuthError {
  let rule = match self.failure_rule {
    1 => LunaApiAuthPolicy
    2 => LunaApiAuthCredential
    3 => LunaApiAuthSource
    4 => LunaApiAuthWorkUnits
    5 => LunaApiAuthLifecycle
    _ => LunaApiAuthEpoch
  }
  let issue = match self.failure_issue {
    1 => LunaApiAuthInvalid
    2 => LunaApiAuthEmpty
    3 => LunaApiAuthTooLong
    4 => LunaApiAuthOutOfRange
    5 => LunaApiAuthBusy
    6 => LunaApiAuthNotReady
    7 => LunaApiAuthStale
    _ => LunaApiAuthExhausted
  }
  luna_api_auth_error(rule, issue)
}

///|
fn LunaApiAuthVerifierWorkspace::progress_one(
  self : LunaApiAuthVerifierWorkspace,
) -> Unit raise LunaApiAuthError {
  if !self.policy.is_authenticated() {
    raise luna_api_auth_error(LunaApiAuthPolicy, LunaApiAuthInvalid)
  }
  if self.cursor < self.policy.expected_length {
    self.difference = self.policy.compare_byte(self.cursor, 0, self.difference)
    self.cursor += 1
  } else {
    self.phase = if self.difference == 0 {
      LUNA_API_AUTH_ACCEPTED
    } else {
      LUNA_API_AUTH_REJECTED
    }
  }
}

///|
pub fn LunaApiAuthVerificationWork::progress(
  self : LunaApiAuthVerificationWork,
) -> LunaApiAuthVerificationProgress raise LunaApiAuthError {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  if workspace.phase == LUNA_API_AUTH_ACCEPTED ||
    workspace.phase == LUNA_API_AUTH_REJECTED ||
    workspace.phase == LUNA_API_AUTH_FAILED {
    return LunaApiAuthVerificationIdle
  }
  let budget = workspace.budget.work_units()
  let mut used = 0
  while used < budget && workspace.phase == LUNA_API_AUTH_PADDING {
    workspace.progress_one() catch {
      error => workspace.fail(error)
    }
    used += 1
  }
  workspace.last_work_units = used
  workspace.total_work_units += used.to_uint64()
  LunaApiAuthVerificationAdvanced
}

///|
pub fn LunaApiAuthVerificationWork::state(
  self : LunaApiAuthVerificationWork,
) -> LunaApiAuthVerificationState raise LunaApiAuthError {
  match self.require_workspace().phase {
    LUNA_API_AUTH_ACCEPTED => LunaApiAuthVerificationAccepted
    LUNA_API_AUTH_REJECTED => LunaApiAuthVerificationRejected
    LUNA_API_AUTH_FAILED => LunaApiAuthVerificationFailed
    _ => LunaApiAuthVerificationPending
  }
}

///|
pub fn LunaApiAuthVerificationWork::last_work_units(
  self : LunaApiAuthVerificationWork,
) -> Int raise LunaApiAuthError {
  self.require_workspace().last_work_units
}

///|
pub fn LunaApiAuthVerificationWork::total_work_units(
  self : LunaApiAuthVerificationWork,
) -> UInt64 raise LunaApiAuthError {
  self.require_workspace().total_work_units
}

///|
pub fn LunaApiAuthVerificationWork::failure(
  self : LunaApiAuthVerificationWork,
) -> LunaApiAuthError raise LunaApiAuthError {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_API_AUTH_FAILED {
    raise luna_api_auth_error(LunaApiAuthLifecycle, LunaApiAuthNotReady)
  }
  workspace.failure_value()
}

///|
pub fn LunaApiAuthVerificationWork::take_decision(
  self : LunaApiAuthVerificationWork,
) -> LunaApiAuthDecision raise LunaApiAuthError {
  let workspace = self.require_workspace()
  let state = match workspace.phase {
    LUNA_API_AUTH_ACCEPTED => 0
    LUNA_API_AUTH_REJECTED => 1
    _ => raise luna_api_auth_error(LunaApiAuthLifecycle, LunaApiAuthNotReady)
  }
  workspace.reset_operation()
  workspace.phase = LUNA_API_AUTH_IDLE
  { state, }
}

///|
pub fn LunaApiAuthVerificationWork::abort(
  self : LunaApiAuthVerificationWork,
) -> Unit raise LunaApiAuthError {
  let workspace = self.require_workspace()
  workspace.reset_operation()
  workspace.phase = LUNA_API_AUTH_IDLE
}
