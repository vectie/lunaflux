///|
fn Scheduler::streaming_request_live(
  self : Scheduler,
  owner : StreamingOwner,
) -> Bool {
  owner.request_slot >= 0 &&
  self.slot_states[owner.request_slot] == SLOT_WAITING &&
  self.slot_generations[owner.request_slot] == owner.request_generation &&
  self.deadlines[owner.request_slot] > owner.last_now
}

///|
fn Scheduler::prepare_streaming_restore(
  self : Scheduler,
  owner : StreamingOwner,
) -> Bool {
  let slot = self.waiting_head
  if slot < 0 || self.prefix_candidate_entry_indices[slot] < 0 {
    return false
  }
  let request = self.requests[slot].unwrap()
  let copied = self.copy_prefix_tokens(request)
  let lookup = self.prefix_lookup.unwrap()
  try! self.prefix_index
  .unwrap()
  .lookup(
    try! request.prefix_key(self.capacity.prefix_layout()),
    self.prefix_tokens.unwrap(),
    maximum_reusable_tokens=copied.min(request.input_tokens().length() - 1),
    lookup,
  )
  if !lookup.has_entry() || lookup.fully_resident() {
    return false
  }
  let cold = for i in 0..<lookup.page_count(); count = 0 {
    continue count + (if (try! lookup.host_at(i)).slot >= 0 { 1 } else { 0 })
  } nobreak {
    count
  }
  let slack = if self.deadlines[slot] <= owner.last_now {
    0L
  } else {
    let remaining = self.deadlines[slot] - owner.last_now
    (if remaining > 9223372036854UL { 9223372036854UL } else { remaining }).to_int64() *
    1000000L
  }
  if cold == 0 ||
    cold > owner.cells.length() ||
    self.page_allocator.free_page_count() -
    self.capacity.emergency_decode_page_reserve() <=
    cold ||
    !owner.cost.should_restore(
      queued_ns=0L,
      deadline_slack_ns=slack / cold.to_int64(),
      destination_reserved=true,
      transfer_credit=true,
    ) {
    return false
  }
  let entry = try! lookup.entry_id()
  if (try! self.prefix_index.unwrap().active_reference_count(entry)) >=
    self.prefix_index.unwrap().limits().max_active_references_per_entry() {
    return false
  }
  try! self.page_allocator.allocate_run(cold, self.page_run)
  try! self.prefix_index.unwrap().retain_active(entry)
  owner.entry = Pinned(entry)
  owner.request_slot = slot
  owner.request_generation = self.slot_generations[slot]
  owner.count = 0
  owner.cursor = 0
  for i in 0..<lookup.page_count() {
    let host = try! lookup.host_at(i)
    if host.slot >= 0 {
      owner.hosts[owner.count] = host
      owner.cells[owner.count] = try! lookup.cell_at(i)
      try! owner.pages.write(
        owner.count,
        try! self.page_run.page_at(owner.count),
      )
      owner.count += 1
    }
  }
  true
}

///|
/// Plans one page effect only while no schedule is outstanding. Source and
/// destination page references are acquired before this command can escape.
pub fn Scheduler::next_streaming_command(
  self : Scheduler,
  now_millis : UInt64,
) -> @streaming_ir.CommandOption raise SchedulerError {
  guard self.streaming is Some(owner) else { return None }
  if owner.worker == 0UL {
    return None
  }
  self.observe_streaming_time(now_millis)
  if self.plan_a_sequence != 0UL || self.plan_b_sequence != 0UL {
    return None
  }
  match owner.flight {
    Spilling(command, _, _) => return Some({ ..command, opcode: Poll, })
    Restoring(command) =>
      return Some({
        ..command,
        opcode: if self.streaming_request_live(owner) {
          Poll
        } else {
          Cancel
        },
      })
    Dropping(command) => return Some(command)
    Empty => ()
  }
  if owner.sequence == 0xffffffffffffffffUL {
    raise invalid(Plan, Exhausted)
  }
  if owner.drop_count > 0 {
    owner.started_at = now_millis
    owner.sequence += 1UL
    let command : @streaming_ir.Command = {
      worker: owner.worker,
      sequence: owner.sequence,
      opcode: Evict,
      page: 0,
      page_generation: 1,
      host: owner.drops[owner.drop_count - 1],
    }
    owner.flight = Dropping(command)
    return Some(command)
  }
  if owner.count > 0 && !self.streaming_request_live(owner) {
    self.release_restore_reservations(owner)
  }
  if owner.count == 0 {
    ignore(self.prepare_streaming_restore(owner))
  }
  if owner.count > 0 {
    owner.started_at = now_millis
    owner.sequence += 1UL
    let command = owner.command(
      Restore,
      try! owner.pages.read(owner.cursor),
      owner.hosts[owner.cursor],
    )
    owner.flight = Restoring(command)
    return Some(command)
  }
  // Do not immediately spill the prefix just restored for the oldest waiter.
  if self.waiting_head >= 0 &&
    self.prefix_candidate_entry_indices[self.waiting_head] >= 0 &&
    self.prefix_lookup.unwrap().has_entry() &&
    self.prefix_lookup.unwrap().fully_resident() {
    return None
  }
  if owner.host_used >= owner.capability.slots() ||
    self.page_allocator.free_page_count() >= owner.capability.reserve_pages() {
    return None
  }
  let index = self.prefix_index.unwrap()
  let capacity = index.limits().page_capacity()
  let scan_count = capacity.min(64)
  for offset in 0..<scan_count {
    let cell = (owner.spill_cursor + offset) % capacity
    if index.page_occupied(cell) && (try! index.page_host(cell)).slot < 0 {
      let page = try! index.page_value(cell)
      if (try! self.page_allocator.active_reference_count(page)) == 0 &&
        (try! self.page_allocator.cached_reference_count(page)) == 1 {
        try! self.page_allocator.retain_active(page)
        owner.sequence += 1UL
        owner.started_at = now_millis
        owner.host_used += 1
        let command = owner.command(Spill, page, { slot: -1, generation: 0, })
        owner.flight = Spilling(command, cell, page)
        owner.spill_cursor = (cell + 1) % capacity
        return Some(command)
      }
    }
  }
  owner.spill_cursor = (owner.spill_cursor + scan_count) % capacity
  None
}

///|
/// Refresh before interpreting a completion, including while transport is pending.
pub fn Scheduler::observe_streaming_time(
  self : Scheduler,
  now_millis : UInt64,
) -> Unit raise SchedulerError {
  guard self.streaming is Some(owner) else { return }
  if now_millis < owner.last_now ||
    (
      !(owner.flight is Empty) &&
      now_millis - owner.started_at >
      owner.capability.transfer_timeout_millis().to_uint64()
    ) {
    raise invalid(Plan, Exhausted)
  }
  owner.last_now = now_millis
}
