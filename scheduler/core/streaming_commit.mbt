///|
fn Scheduler::release_restore_reservations(
  self : Scheduler,
  owner : StreamingOwner,
) -> Unit {
  for i in 0..<owner.count {
    try! self.page_allocator.release_active(try! owner.pages.read(i))
    try! owner.pages.clear(i)
  }
  if owner.entry is Pinned(entry) {
    try! self.prefix_index.unwrap().release_active(entry)
  }
  owner.entry = Empty
  owner.count = 0
  owner.cursor = 0
  owner.request_slot = -1
}

///|
/// A completed prefix transaction publishes all destinations together. The
/// matching request remains waiting; ordinary scheduling acquires it later.
fn Scheduler::publish_restored_prefix(
  self : Scheduler,
  owner : StreamingOwner,
) -> Unit {
  let index = self.prefix_index.unwrap()
  for i in 0..<owner.count {
    if !(try! index.page_host(owner.cells[i])).same(owner.hosts[i]) {
      abort("streaming restore authority invariant")
    }
  }
  for i in 0..<owner.count {
    let page = try! owner.pages.read(i)
    try! self.page_allocator.retain_cached(page)
    if !index.restore_page(owner.cells[i], owner.hosts[i], page) {
      abort("streaming restore publication invariant")
    }
    owner.drop(owner.hosts[i])
  }
  self.release_restore_reservations(owner)
}

///|
pub fn Scheduler::accept_streaming_reply(
  self : Scheduler,
  reply : @streaming_ir.Reply,
) -> Unit raise SchedulerError {
  guard self.streaming is Some(owner) else {
    raise invalid(Plan, InvalidTransition)
  }
  let expected = match owner.flight {
    Spilling(command, _, _) | Restoring(command) | Dropping(command) => command
    Empty => raise invalid(Plan, InvalidTransition)
  }
  if !expected.same_transaction(reply.command) || reply.status is Rejected {
    raise invalid(Plan, Mismatch)
  }
  if reply.status is Pending {
    return
  }
  match owner.flight {
    Spilling(_, cell, page) => {
      if !(reply.status is HostReady) ||
        reply.host.slot < 0 ||
        reply.host.slot >= owner.capability.slots() ||
        reply.host.generation <= 0 {
        raise invalid(Plan, Mismatch)
      }
      // Constant-time ownership check; no radix qualification scan on progress.
      let index = self.prefix_index.unwrap()
      if owner.host_live[reply.host.slot] ||
        reply.host.generation <= owner.host_generations[reply.host.slot] {
        raise invalid(Plan, Mismatch)
      }
      if !index.spill_page(cell, page, reply.host) {
        raise invalid(Plan, Mismatch)
      }
      owner.host_live[reply.host.slot] = true
      owner.host_generations[reply.host.slot] = reply.host.generation
      try! self.page_allocator.release_cached(page)
      try! self.page_allocator.release_active(page)
      owner.flight = Empty
    }
    Restoring(_) => {
      if !reply.host.same(owner.hosts[owner.cursor]) {
        raise invalid(Plan, Mismatch)
      }
      if reply.status is Cancelled {
        self.release_restore_reservations(owner)
        owner.flight = Empty
        return
      }
      if !(reply.status is DeviceReady) {
        raise invalid(Plan, Mismatch)
      }
      owner.cursor += 1
      owner.flight = Empty
      if owner.cursor == owner.count {
        if self.streaming_request_live(owner) {
          self.publish_restored_prefix(owner)
        } else {
          self.release_restore_reservations(owner)
        }
      } else if !self.streaming_request_live(owner) {
        // The last submitted page completed; all remaining pages were never
        // submitted, so the entire reservation can now be safely released.
        self.release_restore_reservations(owner)
      }
    }
    Dropping(_) => {
      if !(reply.status is Evicted) || !reply.host.same(expected.host) {
        raise invalid(Plan, Mismatch)
      }
      owner.drop_count -= 1
      owner.host_live[reply.host.slot] = false
      owner.host_used -= 1
      owner.flight = Empty
    }
    Empty => raise invalid(Plan, InvalidTransition)
  }
}

///|
/// Called at the existing device-loss boundary, after the child has been reaped.
/// No transport error or cancellation alone grants this reclamation authority.
fn Scheduler::forget_streaming_worker(self : Scheduler) -> Unit {
  guard self.streaming is Some(owner) else { return }
  match owner.flight {
    Spilling(_, _, page) => try! self.page_allocator.release_active(page)
    _ => ()
  }
  if owner.count > 0 {
    self.release_restore_reservations(owner)
  }
  owner.flight = Empty
  owner.drop_count = 0
  owner.host_used = 0
  owner.sequence = 0UL
  for i in 0..<owner.host_live.length() {
    owner.host_live[i] = false
    owner.host_generations[i] = 0
  }
  // Worker zero prevents any command from escaping before replacement binds.
  owner.worker = 0UL
}

///|
pub fn Scheduler::bind_streaming_replacement(
  self : Scheduler,
  worker : UInt64,
) -> Unit raise SchedulerError {
  guard self.streaming is Some(owner) else { return }
  if owner.worker != 0UL ||
    worker <= owner.last_worker ||
    !(owner.flight is Empty) ||
    owner.host_used != 0 {
    raise invalid(Plan, InvalidTransition)
  }
  owner.worker = worker
  owner.last_worker = worker
}
