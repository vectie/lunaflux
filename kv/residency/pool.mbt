///|
pub fn Pool::new(limits : Limits) -> Pool {
  {
    limits,
    states: FixedArray::make(limits.slots, {
      phase: Free,
      generation: 1,
      epoch: 0,
      cancelled: false,
    }),
    free: FixedArray::makei(limits.slots, i => limits.slots - i - 1),
    free_count: limits.slots,
    inflight: 0,
    closed: false,
  }
}

///|
fn Pool::owns(self : Pool, entry : Entry) -> Bool {
  physical_equal(self, entry.owner) &&
  entry.slot >= 0 &&
  entry.slot < self.limits.slots &&
  self.states[entry.slot].generation == entry.generation
}

///|
fn Pool::commit(self : Pool, slot : Int, event : Event) -> Effect {
  let before = self.states[slot]
  let decision = transition(before, event)
  self.states[slot] = decision.state
  if pending(before.phase) && !pending(decision.state.phase) {
    self.inflight -= 1
  } else if !pending(before.phase) && pending(decision.state.phase) {
    self.inflight += 1
  }
  if decision.effect is Released && decision.state.phase is Free {
    self.free[self.free_count] = slot
    self.free_count += 1
  }
  decision.effect
}

///|
/// Reserves storage and a transfer credit atomically before any device effect.
pub fn Pool::begin_spill(self : Pool) -> Begin {
  if self.closed {
    return Closed
  }
  if self.free_count == 0 || self.inflight == self.limits.transfers {
    return Backpressured
  }
  let slot = self.free[self.free_count - 1]
  self.free_count -= 1
  let _ = self.commit(slot, BeginSpill)
  Started({
    entry: { owner: self, slot, generation: self.states[slot].generation, },
    epoch: self.states[slot].epoch,
  })
}

///|
pub fn Pool::begin_restore(self : Pool, entry : Entry) -> Begin {
  if self.closed {
    return Closed
  }
  if !self.owns(entry) ||
    !(self.states[entry.slot].phase is Host) ||
    self.states[entry.slot].epoch == 2147483647 {
    return InvalidEntry
  }
  if self.inflight == self.limits.transfers {
    return Backpressured
  }
  let _ = self.commit(entry.slot, BeginRestore)
  Started({ entry, epoch: self.states[entry.slot].epoch, })
}

///|
fn Pool::apply(self : Pool, transfer : Transfer, event : Event) -> Effect {
  if self.closed ||
    !self.owns(transfer.entry) ||
    self.states[transfer.entry.slot].epoch != transfer.epoch {
    return Rejected
  }
  self.commit(transfer.entry.slot, event)
}

///|
pub fn Pool::cancel(self : Pool, transfer : Transfer) -> Effect {
  self.apply(transfer, Cancel)
}

///|
/// Called by the effect interpreter only after successful native completion.
pub fn Pool::complete(self : Pool, transfer : Transfer) -> Effect {
  self.apply(transfer, Complete)
}

///|
pub fn Pool::fail(self : Pool, transfer : Transfer) -> Effect {
  self.apply(transfer, Fail)
}

///|
/// A failed submit/poll is not a drain. Reservations survive until the native
/// transfer owner has proved that no DMA can still access either endpoint.
pub fn Pool::drained(self : Pool, transfer : Transfer) -> Effect {
  self.apply(transfer, Drained)
}

///|
pub fn Pool::evict(self : Pool, entry : Entry) -> Effect {
  if self.closed || !self.owns(entry) {
    return Rejected
  }
  self.commit(entry.slot, Evict)
}

///|
pub fn Pool::is_host_ready(self : Pool, entry : Entry) -> Bool {
  !self.closed && self.owns(entry) && self.states[entry.slot].phase is Host
}

///|
pub fn Pool::inflight(self : Pool) -> Int {
  self.inflight
}

///|
pub fn Pool::available(self : Pool) -> Int {
  self.free_count
}

///|
/// Closing never cancels DMA or silently releases its reservations.
pub fn Pool::close(self : Pool) -> Bool {
  if self.inflight != 0 {
    return false
  }
  self.closed = true
  true
}
