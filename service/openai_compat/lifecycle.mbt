///|
fn LunaOpenAICompatWork::require_workspace(
  self : LunaOpenAICompatWork,
) -> LunaOpenAICompatWorkspace raise LunaOpenAICompatError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase == LUNA_OPENAI_COMPAT_IDLE ||
    self.workspace.phase == LUNA_OPENAI_COMPAT_VIEW_LEASED {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatEpoch)
  }
  self.workspace
}

///|
fn LunaOpenAICompatWorkspace::fail(
  self : LunaOpenAICompatWorkspace,
  error : LunaOpenAICompatError,
) -> Unit {
  let LunaOpenAICompatInvalid(rule) = error
  self.failure_rule = match rule {
    LunaOpenAICompatLimits => 1
    LunaOpenAICompatWorkUnits => 2
    LunaOpenAICompatModelAlias => 3
    LunaOpenAICompatResponseId => 4
    LunaOpenAICompatLifecycle => 5
    LunaOpenAICompatEpoch => 6
    LunaOpenAICompatSemanticEvent => 7
    LunaOpenAICompatScalar => 8
    LunaOpenAICompatPayload => 9
    LunaOpenAICompatCapacity => 10
    LunaOpenAICompatDestination => 11
    LunaOpenAICompatJson => 12
    LunaOpenAICompatTemplate => 13
    LunaOpenAICompatHttp => 14
    LunaOpenAICompatFrame => 15
  }
  self.phase = LUNA_OPENAI_COMPAT_FAILED
}

///|
fn LunaOpenAICompatWorkspace::failure_value(
  self : LunaOpenAICompatWorkspace,
) -> LunaOpenAICompatError {
  LunaOpenAICompatInvalid(
    match self.failure_rule {
      1 => LunaOpenAICompatLimits
      2 => LunaOpenAICompatWorkUnits
      3 => LunaOpenAICompatModelAlias
      4 => LunaOpenAICompatResponseId
      5 => LunaOpenAICompatLifecycle
      6 => LunaOpenAICompatEpoch
      7 => LunaOpenAICompatSemanticEvent
      8 => LunaOpenAICompatScalar
      9 => LunaOpenAICompatPayload
      10 => LunaOpenAICompatCapacity
      11 => LunaOpenAICompatDestination
      12 => LunaOpenAICompatJson
      13 => LunaOpenAICompatTemplate
      14 => LunaOpenAICompatHttp
      _ => LunaOpenAICompatFrame
    },
  )
}

///|
fn LunaOpenAICompatWorkspace::progress_one(
  self : LunaOpenAICompatWorkspace,
) -> Unit raise LunaOpenAICompatError {
  let event = self.authenticate_event()
  match self.phase {
    LUNA_OPENAI_COMPAT_SETUP => self.progress_setup(event)
    LUNA_OPENAI_COMPAT_CONFIGURE_PART => self.configure_part()
    LUNA_OPENAI_COMPAT_LITERAL => self.progress_literal()
    LUNA_OPENAI_COMPAT_RAW => self.progress_raw()
    LUNA_OPENAI_COMPAT_LOAD_ESCAPE => self.progress_load_escape(event)
    LUNA_OPENAI_COMPAT_EMIT_ESCAPE => self.progress_emit_escape()
    LUNA_OPENAI_COMPAT_DECIMAL_PREPARE => self.progress_decimal_prepare()
    LUNA_OPENAI_COMPAT_DECIMAL_EMIT => self.progress_decimal_emit()
    LUNA_OPENAI_COMPAT_FINALIZING => {
      // authenticate_event above is the final semantic liveness proof. The
      // completed SSE frame is independent and carries no ACK authority.
      self.semantic_views.clear()
      self.phase = LUNA_OPENAI_COMPAT_READY
    }
    _ => abort("Luna OpenAI compatibility progress phase invariant")
  }
}

///|
pub fn LunaOpenAICompatWork::progress(
  self : LunaOpenAICompatWork,
) -> LunaOpenAICompatProgress raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  if workspace.phase == LUNA_OPENAI_COMPAT_READY ||
    workspace.phase == LUNA_OPENAI_COMPAT_FAILED {
    return LunaOpenAICompatIdle
  }
  let budget = workspace.budget.work_units()
  let mut used = 0
  while used < budget &&
        workspace.phase != LUNA_OPENAI_COMPAT_READY &&
        workspace.phase != LUNA_OPENAI_COMPAT_FAILED {
    workspace.progress_one() catch {
      error => workspace.fail(error)
    }
    used += 1
  }
  workspace.last_work_units = used
  workspace.total_work_units += used.to_uint64()
  LunaOpenAICompatAdvanced
}

///|
pub fn LunaOpenAICompatWork::state(
  self : LunaOpenAICompatWork,
) -> LunaOpenAICompatWorkState raise LunaOpenAICompatError {
  match self.require_workspace().phase {
    LUNA_OPENAI_COMPAT_READY => LunaOpenAICompatReady
    LUNA_OPENAI_COMPAT_FAILED => LunaOpenAICompatFailed
    _ => LunaOpenAICompatPending
  }
}

///|
pub fn LunaOpenAICompatWork::last_work_units(
  self : LunaOpenAICompatWork,
) -> Int raise LunaOpenAICompatError {
  self.require_workspace().last_work_units
}

///|
pub fn LunaOpenAICompatWork::total_work_units(
  self : LunaOpenAICompatWork,
) -> UInt64 raise LunaOpenAICompatError {
  self.require_workspace().total_work_units
}

///|
pub fn LunaOpenAICompatWork::failure(
  self : LunaOpenAICompatWork,
) -> LunaOpenAICompatError raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_OPENAI_COMPAT_FAILED {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLifecycle)
  }
  workspace.failure_value()
}

///|
pub fn LunaOpenAICompatWork::take_view(
  self : LunaOpenAICompatWork,
) -> LunaOpenAICompatView raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_OPENAI_COMPAT_READY {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLifecycle)
  }
  workspace.phase = LUNA_OPENAI_COMPAT_VIEW_LEASED
  { workspace, epoch: self.epoch }
}

///|
pub fn LunaOpenAICompatWork::abort(
  self : LunaOpenAICompatWork,
) -> Unit raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  workspace.reset_operation()
  workspace.phase = LUNA_OPENAI_COMPAT_IDLE
}

///|
fn LunaOpenAICompatView::require_workspace(
  self : LunaOpenAICompatView,
) -> LunaOpenAICompatWorkspace raise LunaOpenAICompatError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_OPENAI_COMPAT_VIEW_LEASED {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatEpoch)
  }
  self.workspace
}

///|
pub fn LunaOpenAICompatView::length(
  self : LunaOpenAICompatView,
) -> Int raise LunaOpenAICompatError {
  self.require_workspace().length
}

///|
pub fn LunaOpenAICompatView::copy_chunk_to(
  self : LunaOpenAICompatView,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  source_offset~ : Int,
  length~ : Int,
) -> Int raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  if length <= 0 || length > workspace.budget.work_units() {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatDestination)
  }
  if source_offset < 0 ||
    source_offset > workspace.length ||
    length > workspace.length - source_offset ||
    destination_offset < 0 ||
    destination_offset > destination.length() ||
    length > destination.length() - destination_offset {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatDestination)
  }
  for index in 0..<length {
    destination[destination_offset + index] = workspace.storage[source_offset +
      index]
  }
  length
}

///|
pub fn LunaOpenAICompatView::release(
  self : LunaOpenAICompatView,
) -> Unit raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  workspace.reset_operation()
  workspace.phase = LUNA_OPENAI_COMPAT_IDLE
}
