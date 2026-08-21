///|
pub fn LunaFramedEventAdapter::new(
  limits : FramedWireLimits,
) -> LunaFramedEventAdapter raise FramedWireError {
  {
    workspace: LunaFramedEventWorkspace::new(
      limits,
      try! LunaFramedEventStepBudget::new(LUNA_FRAMED_EVENT_MAX_STEP_WORK_UNITS),
    ),
    views: Array::new(capacity=1),
  }
}

///|
/// Synchronously drives the same authoritative encoder used by bounded
/// transport owners. This compatibility entry point is proportional to the
/// encoded event length and is not a reactor-quantum API.
pub fn LunaFramedEventAdapter::stage(
  self : LunaFramedEventAdapter,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  if !self.views.is_empty() {
    raise EventCreditUnavailable
  }
  let work = self.workspace.begin(event)
  while work.state() == LunaFramedEventPending {
    ignore(work.progress())
  }
  if work.state() == LunaFramedEventFailed {
    let error = work.failure()
    work.abort()
    raise error
  }
  self.views.push(work.take_view())
}

///|
pub fn LunaFramedEventAdapter::length(
  self : LunaFramedEventAdapter,
) -> Int raise FramedWireError {
  if self.views.length() != 1 {
    raise StaleFrame
  }
  self.views[0].length()
}

///|
pub fn LunaFramedEventAdapter::copy_to(
  self : LunaFramedEventAdapter,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise FramedWireError {
  let length = self.length()
  if destination_offset < 0 ||
    destination_offset > destination.length() - length {
    raise DestinationTooSmall
  }
  let view = self.views[0]
  let quantum = self.workspace.budget.work_units()
  let mut cursor = 0
  while cursor < length {
    let amount = Int::min(quantum, length - cursor)
    ignore(
      view.copy_chunk_to(
        destination,
        destination_offset=destination_offset + cursor,
        source_offset=cursor,
        length=amount,
      ),
    )
    cursor += amount
  }
  length
}

///|
/// Releases only the adapter's framed-byte credit. It never retires or
/// acknowledges the semantic event view.
pub fn LunaFramedEventAdapter::release(
  self : LunaFramedEventAdapter,
) -> Unit raise FramedWireError {
  if self.views.length() != 1 {
    raise StaleFrame
  }
  self.views[0].release()
  self.views.clear()
}
