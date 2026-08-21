///|
pub fn LunaFramedEventWork::take_view(
  self : LunaFramedEventWork,
) -> LunaFramedEventView raise FramedWireError {
  let workspace = self.require_workspace()
  if workspace.phase == LUNA_FRAMED_EVENT_FAILED {
    raise LunaFramedEventFailed
  }
  if workspace.phase != LUNA_FRAMED_EVENT_READY {
    raise LunaFramedEventNotReady
  }
  workspace.phase = LUNA_FRAMED_EVENT_VIEW_LEASED
  { workspace, epoch: self.epoch }
}

///|
fn LunaFramedEventView::require_workspace(
  self : LunaFramedEventView,
) -> LunaFramedEventWorkspace raise FramedWireError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_FRAMED_EVENT_VIEW_LEASED {
    raise LunaFramedEventStale
  }
  self.workspace
}

///|
pub fn LunaFramedEventView::length(
  self : LunaFramedEventView,
) -> Int raise FramedWireError {
  self.require_workspace().length
}

///|
/// Copies exactly one caller-selected chunk. The configured step budget caps
/// the chunk, while the transport retains its own confirmed-write cursor.
pub fn LunaFramedEventView::copy_chunk_to(
  self : LunaFramedEventView,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  source_offset~ : Int,
  length~ : Int,
) -> Int raise FramedWireError {
  let workspace = self.require_workspace()
  if length <= 0 ||
    length > workspace.budget.work_units() ||
    source_offset < 0 ||
    source_offset > workspace.length - length {
    raise LunaFramedEventInvalidRange
  }
  if destination_offset < 0 ||
    destination_offset > destination.length() - length {
    raise DestinationTooSmall
  }
  for index in 0..<length {
    destination[destination_offset + index] = workspace.storage[source_offset +
      index]
  }
  length
}

///|
/// Releases only immutable framed-byte authority. It never acknowledges the
/// semantic event from which the frame was constructed.
pub fn LunaFramedEventView::release(
  self : LunaFramedEventView,
) -> Unit raise FramedWireError {
  let workspace = self.require_workspace()
  workspace.phase = LUNA_FRAMED_EVENT_IDLE
  workspace.length = 0
  workspace.payload_length = 0
  workspace.cursor = 0
}
