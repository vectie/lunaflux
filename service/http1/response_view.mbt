///|
fn LunaHttp1ResponseView::require_workspace(
  self : LunaHttp1ResponseView,
) -> LunaHttp1ResponseWorkspace raise LunaHttp1Error {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_HTTP1_RESPONSE_VIEW_LEASED {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Stale)
  }
  self.workspace
}

///|
pub fn LunaHttp1ResponseView::is_live(self : LunaHttp1ResponseView) -> Bool {
  self.epoch != 0UL &&
  self.workspace.epoch == self.epoch &&
  self.workspace.phase == LUNA_HTTP1_RESPONSE_VIEW_LEASED
}

///|
pub fn LunaHttp1ResponseView::status_code(
  self : LunaHttp1ResponseView,
) -> Int raise LunaHttp1Error {
  self.require_workspace().status_code
}

///|
pub fn LunaHttp1ResponseView::length(
  self : LunaHttp1ResponseView,
) -> Int raise LunaHttp1Error {
  self.require_workspace().response_length
}

///|
pub fn LunaHttp1ResponseView::byte_at(
  self : LunaHttp1ResponseView,
  index : Int,
) -> Byte raise LunaHttp1Error {
  let workspace = self.require_workspace()
  if index < 0 || index >= workspace.response_length {
    raise luna_http1_error(LunaHttp1Source, LunaHttp1OutOfRange)
  }
  workspace.storage[index]
}

///|
/// Copies at most one configured response quantum into caller-owned storage.
pub fn LunaHttp1ResponseView::copy_chunk(
  self : LunaHttp1ResponseView,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  source_offset~ : Int,
  length~ : Int,
) -> Int raise LunaHttp1Error {
  let workspace = self.require_workspace()
  if source_offset < 0 ||
    source_offset > workspace.response_length ||
    length < 0 {
    raise luna_http1_error(LunaHttp1Source, LunaHttp1OutOfRange)
  }
  let amount = Int::min(
    length,
    Int::min(
      workspace.budget.work_units(),
      workspace.response_length - source_offset,
    ),
  )
  if destination_offset < 0 ||
    destination_offset > destination.length() ||
    amount > destination.length() - destination_offset {
    raise luna_http1_error(LunaHttp1Source, LunaHttp1OutOfRange)
  }
  for index in 0..<amount {
    destination[destination_offset + index] = workspace.storage[source_offset +
      index]
  }
  amount
}

///|
pub fn LunaHttp1ResponseView::release(
  self : LunaHttp1ResponseView,
) -> Unit raise LunaHttp1Error {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  workspace.phase = LUNA_HTTP1_RESPONSE_IDLE
}
