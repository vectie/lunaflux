///|
fn LunaOpenAIInboundWork::require_workspace(
  self : Self,
) -> LunaOpenAIInboundWorkspace raise LunaOpenAICompatError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase == LUNA_INBOUND_IDLE ||
    self.workspace.phase == LUNA_INBOUND_VIEW {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatEpoch)
  }
  self.workspace
}

///|
fn LunaOpenAIInboundWorkspace::authenticate_http(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.http_views.length() != 1 ||
    !self.http_views[0].is_live() ||
    (self.http_views[0].body_length() catch { _ => -1 }) != self.body_length {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatHttp)
  }
}

///|
fn LunaOpenAIInboundWorkspace::fail_inbound(
  self : Self,
  error : LunaOpenAICompatError,
) -> Unit {
  let LunaOpenAICompatInvalid(rule) = error
  self.failure_rule = match rule {
    LunaOpenAICompatJson => 1
    LunaOpenAICompatTemplate => 2
    LunaOpenAICompatHttp => 3
    LunaOpenAICompatFrame => 4
    LunaOpenAICompatCapacity => 5
    LunaOpenAICompatEpoch => 6
    LunaOpenAICompatDestination => 7
    _ => 8
  }
  self.cleanup_children()
  self.phase = LUNA_INBOUND_FAILED
}

///|
fn LunaOpenAIInboundWorkspace::inbound_failure(
  self : Self,
) -> LunaOpenAICompatError {
  LunaOpenAICompatInvalid(
    match self.failure_rule {
      1 => LunaOpenAICompatJson
      2 => LunaOpenAICompatTemplate
      3 => LunaOpenAICompatHttp
      4 => LunaOpenAICompatFrame
      5 => LunaOpenAICompatCapacity
      6 => LunaOpenAICompatEpoch
      7 => LunaOpenAICompatDestination
      _ => LunaOpenAICompatLifecycle
    },
  )
}

///|
fn LunaOpenAIInboundWorkspace::begin_handoff(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.prompt_length <= 0 || self.handoff_writes.length() != 1 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  self.handoff_push_stage = 1
  self.handoff_stop_index = 0
  self.handoff_stop_byte = -1
  self.phase = LUNA_INBOUND_HANDOFF_PUSH
}

///|
fn LunaOpenAIInboundWorkspace::progress_handoff_push(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.handoff_push_stage == 1 &&
    self.handoff_stop_index < self.stop_token_count {
    self.handoff_writes[0].push_stop_token(
      self.stop_tokens[self.handoff_stop_index],
    ) catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.handoff_stop_index += 1
    return
  }
  if self.handoff_push_stage == 1 {
    self.handoff_push_stage = 2
    self.handoff_stop_index = 0
    return
  }
  if self.handoff_push_stage == 2 &&
    self.handoff_stop_index < self.stop_string_count {
    let base = self.handoff_stop_index * 2
    if self.handoff_stop_byte < 0 {
      self.handoff_writes[0].push_stop_string_length(
        self.stop_string_cells[base + 1],
      ) catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.handoff_stop_byte = 0
      return
    }
    if self.handoff_stop_byte < self.stop_string_cells[base + 1] {
      self.handoff_writes[0].push_stop_string_byte(
        self.stop_string_bytes[self.stop_string_cells[base] +
        self.handoff_stop_byte],
      ) catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.handoff_stop_byte += 1
      return
    }
    self.handoff_stop_index += 1
    self.handoff_stop_byte = -1
    return
  }
  let work = self.handoff_writes[0].finish() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  self.handoff_writes.clear()
  if self.handoff_works.length() != 0 {
    abort("Luna OpenAI handoff work slot invariant")
  }
  self.handoff_works.push(work)
  self.phase = LUNA_INBOUND_HANDOFF_PROGRESS
}

///|
fn LunaOpenAIInboundWorkspace::progress_handoff(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  let work = self.handoff_works[0]
  let progress = work.progress() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  if progress.is_failed() {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  if progress.is_ready() {
    let lease = work.take_lease() catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.handoff_works.clear()
    if self.handoff_leases.length() != 0 {
      abort("Luna OpenAI handoff lease slot invariant")
    }
    self.handoff_leases.push(lease)
    self.phase = LUNA_INBOUND_FINAL
  }
}

///|
fn LunaOpenAIInboundWorkspace::progress_inbound_one(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  self.authenticate_http()
  match self.phase {
    LUNA_INBOUND_STRING => self.progress_string()
    LUNA_INBOUND_STRING_EMIT => self.progress_string_emit()
    LUNA_INBOUND_TRUE => self.progress_true()
    LUNA_INBOUND_NUMBER => self.progress_number()
    LUNA_INBOUND_NUMBER_SCALE => self.progress_number_scale()
    LUNA_INBOUND_NUMBER_FINAL => self.progress_number_final()
    LUNA_INBOUND_RENDER_SEGMENT => self.progress_render_segment()
    LUNA_INBOUND_RENDER_STRING => self.progress_render_string()
    LUNA_INBOUND_RENDER_EMIT => self.progress_render_emit()
    LUNA_INBOUND_HANDOFF_BEGIN => self.begin_handoff()
    LUNA_INBOUND_HANDOFF_PUSH => self.progress_handoff_push()
    LUNA_INBOUND_HANDOFF_PROGRESS => self.progress_handoff()
    LUNA_INBOUND_FINAL => {
      // The authenticated HTTP body remains authoritative through publication.
      self.http_views[0].release() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatHttp)
      }
      self.http_views.clear()
      self.request_ids.clear()
      self.phase = LUNA_INBOUND_READY
    }
    _ => self.progress_parse()
  }
}

///|
pub fn LunaOpenAIInboundWork::progress(
  self : Self,
) -> LunaOpenAIInboundProgress raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  if workspace.phase == LUNA_INBOUND_READY ||
    workspace.phase == LUNA_INBOUND_FAILED {
    return LunaOpenAIInboundIdle
  }
  let mut used = 0
  while used < workspace.budget.work_units() &&
        workspace.phase != LUNA_INBOUND_READY &&
        workspace.phase != LUNA_INBOUND_FAILED {
    workspace.progress_inbound_one() catch {
      error => workspace.fail_inbound(error)
    }
    used += 1
  }
  workspace.last_work_units = used
  workspace.total_work_units += used.to_uint64()
  LunaOpenAIInboundAdvanced
}

///|
pub fn LunaOpenAIInboundWork::state(
  self : Self,
) -> LunaOpenAIInboundWorkState raise LunaOpenAICompatError {
  match self.require_workspace().phase {
    LUNA_INBOUND_READY => LunaOpenAIInboundReady
    LUNA_INBOUND_FAILED => LunaOpenAIInboundFailed
    _ => LunaOpenAIInboundPending
  }
}

///|
pub fn LunaOpenAIInboundWork::last_work_units(
  self : Self,
) -> Int raise LunaOpenAICompatError {
  self.require_workspace().last_work_units
}

///|
pub fn LunaOpenAIInboundWork::total_work_units(
  self : Self,
) -> UInt64 raise LunaOpenAICompatError {
  self.require_workspace().total_work_units
}

///|
pub fn LunaOpenAIInboundWork::failure(
  self : Self,
) -> LunaOpenAICompatError raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_INBOUND_FAILED {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLifecycle)
  }
  workspace.inbound_failure()
}

///|
fn LunaOpenAIInboundWorkspace::cleanup_children(self : Self) -> Unit {
  if self.handoff_writes.length() == 1 {
    self.handoff_writes[0].abort() catch {
      _ => abort("Luna OpenAI handoff write cleanup invariant")
    }
    self.handoff_writes.clear()
  }
  if self.handoff_works.length() == 1 {
    self.handoff_works[0].abort() catch {
      _ => abort("Luna OpenAI handoff work cleanup invariant")
    }
    self.handoff_works.clear()
  }
  if self.handoff_leases.length() == 1 {
    self.handoff_leases[0].release() catch {
      _ => abort("Luna OpenAI handoff lease cleanup invariant")
    }
    self.handoff_leases.clear()
  }
  if self.http_views.length() == 1 {
    if self.http_views[0].is_live() {
      self.http_views[0].release() catch {
        _ => abort("Luna OpenAI HTTP cleanup invariant")
      }
    }
    self.http_views.clear()
  }
}

///|
pub fn LunaOpenAIInboundWork::abort(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  workspace.cleanup_children()
  workspace.reset_operation()
  workspace.phase = LUNA_INBOUND_IDLE
}

///|
pub fn LunaOpenAIInboundWork::take_view(
  self : Self,
) -> LunaOpenAIInboundView raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_INBOUND_READY ||
    workspace.handoff_leases.length() != 1 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLifecycle)
  }
  workspace.phase = LUNA_INBOUND_VIEW
  { workspace, epoch: self.epoch }
}

///|
fn LunaOpenAIInboundView::require_workspace(
  self : Self,
) -> LunaOpenAIInboundWorkspace raise LunaOpenAICompatError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_INBOUND_VIEW ||
    self.workspace.handoff_leases.length() != 1 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatEpoch)
  }
  self.workspace
}

///|
pub fn LunaOpenAIInboundView::handoff(
  self : Self,
) -> @inference.LunaTextRequestHandoffLease raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  workspace.handoff_leases[0]
}

///|
pub fn LunaOpenAIInboundView::length(
  self : Self,
) -> Int raise LunaOpenAICompatError {
  let lease = self.handoff()
  lease.prompt_length() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatFrame)
  }
}

///|
pub fn LunaOpenAIInboundView::copy_chunk_to(
  self : Self,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  source_offset~ : Int,
  length~ : Int,
) -> Int raise LunaOpenAICompatError {
  let lease = self.handoff()
  let total = lease.prompt_length() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatFrame)
  }
  if length <= 0 ||
    length > self.require_workspace().budget.work_units() ||
    source_offset < 0 ||
    source_offset > total - length ||
    destination_offset < 0 ||
    destination_offset > destination.length() - length {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatDestination)
  }
  for index in 0..<length {
    destination[destination_offset + index] = lease.prompt_byte_at(
      source_offset + index,
    ) catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatFrame)
    }
  }
  length
}

///|
pub fn LunaOpenAIInboundView::release_after_transfer(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  workspace.handoff_leases.clear()
  workspace.reset_operation()
  workspace.phase = LUNA_INBOUND_IDLE
}

///|
pub fn LunaOpenAIInboundView::release(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  let workspace = self.require_workspace()
  workspace.handoff_leases[0].release() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatFrame)
  }
  workspace.handoff_leases.clear()
  workspace.reset_operation()
  workspace.phase = LUNA_INBOUND_IDLE
}
