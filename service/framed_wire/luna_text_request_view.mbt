///|
pub fn LunaFramedTextRequestWork::take_view(
  self : Self,
) -> LunaFramedTextRequestView raise FramedWireError {
  let workspace = self.require()
  if workspace.phase != LUNA_TEXT_READY {
    raise LunaFramedTextRequestNotReady
  }
  workspace.phase = LUNA_TEXT_VIEW
  { workspace, epoch: self.epoch }
}

///|
fn LunaFramedTextRequestView::require(
  self : Self,
) -> LunaFramedTextRequestWorkspace raise FramedWireError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_TEXT_VIEW {
    raise LunaFramedTextRequestStale
  }
  self.workspace
}

///|
pub fn LunaFramedTextRequestView::length(
  self : Self,
) -> Int raise FramedWireError {
  let workspace = self.require()
  workspace.payload_cursor + workspace.cache_scope_length
}

///|
pub fn LunaFramedTextRequestView::copy_chunk_to(
  self : Self,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  source_offset~ : Int,
  length~ : Int,
) -> Int raise FramedWireError {
  let workspace = self.require()
  let total = workspace.payload_cursor + workspace.cache_scope_length
  if length <= 0 ||
    length > workspace.budget.work_units() ||
    source_offset < 0 ||
    source_offset > total ||
    length > total - source_offset ||
    destination_offset < 0 ||
    destination_offset > destination.length() ||
    length > destination.length() - destination_offset {
    raise LunaFramedTextRequestInvalidRange
  }
  for index in 0..<length {
    destination[destination_offset + index] = workspace.storage[source_offset +
      index]
  }
  length
}

///|
pub fn LunaFramedTextRequestView::release(
  self : Self,
) -> Unit raise FramedWireError {
  let workspace = self.require()
  workspace.reset()
  workspace.phase = LUNA_TEXT_IDLE
}
