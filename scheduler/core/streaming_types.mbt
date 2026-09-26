///|
#valtype
priv enum StreamingFlight {
  Empty
  Spilling(@streaming_ir.Command, Int, @page_allocator.PageId)
  Restoring(@streaming_ir.Command)
  Dropping(@streaming_ir.Command)
}

///|
#valtype
priv enum StreamingEntry {
  Empty
  Pinned(@radix.PrefixEntryId)
}

///|
priv struct StreamingOwner {
  capability : @streaming_ir.Capability
  cost : @streaming_policy.RestoreCost
  shapes : FixedArray[@streaming_policy.PrefillShape]
  mut worker : UInt64
  mut last_worker : UInt64
  mut sequence : UInt64
  mut flight : StreamingFlight
  drops : FixedArray[@streaming_ir.HostPage]
  mut drop_count : Int
  mut host_used : Int
  host_live : FixedArray[Bool]
  host_generations : FixedArray[Int]
  mut last_now : UInt64
  mut started_at : UInt64
  cells : FixedArray[Int]
  hosts : FixedArray[@streaming_ir.HostPage]
  pages : @page_allocator.PageIdStorage
  mut count : Int
  mut cursor : Int
  mut request_slot : Int
  mut request_generation : Int
  mut entry : StreamingEntry
  mut spill_cursor : Int
}

///|
pub fn Scheduler::configure_streaming(
  self : Scheduler,
  capability : @streaming_ir.Capability,
  worker : UInt64,
) -> Unit raise SchedulerError {
  if !self.capacity.prefix_enabled() ||
    self.streaming is Some(_) ||
    worker == 0UL ||
    capability.page_count() != self.page_allocator.limits().page_capacity() ||
    self.live_count != 0 ||
    self.last_plan_sequence != self.last_retired_plan_sequence {
    raise invalid(PageCapacity, InvalidTransition)
  }
  let count = self.capacity.block_table_pages_per_request()
  let shapes = FixedArray::makei(capability.shape_count(), i => {
    let shape = capability.shape_at(i)
    try! @streaming_policy.PrefillShape::new(
      tokens=shape.tokens(),
      workspace_bytes=shape.workspace_bytes(),
    )
  })
  if capability.shape_count() > 0 {
    let mut has_one = false
    for i in 0..<capability.shape_count() {
      let shape = capability.shape_at(i)
      if shape.tokens() > self.capacity.prefill_chunk_tokens() {
        raise invalid(PageCapacity, OutOfRange)
      }
      if shape.tokens() == 1 &&
        shape.workspace_bytes() <= capability.workspace_budget() {
        has_one = true
      }
    }
    if !has_one {
      raise invalid(PageCapacity, OutOfRange)
    }
  }
  self.streaming = Some({
    capability,
    shapes,
    cost: try! @streaming_policy.RestoreCost::new(
      restore_ns=capability.restore_ns(),
      recompute_ns=capability.recompute_ns(),
      minimum_saving_ns=0L,
    ),
    worker,
    last_worker: worker,
    sequence: 0UL,
    flight: Empty,
    drops: FixedArray::make(capability.slots(), { slot: -1, generation: 0, }),
    drop_count: 0,
    host_used: 0,
    host_live: FixedArray::make(capability.slots(), false),
    host_generations: FixedArray::make(capability.slots(), 0),
    last_now: 0UL,
    started_at: 0UL,
    cells: FixedArray::make(count, -1),
    hosts: FixedArray::make(count, { slot: -1, generation: 0, }),
    pages: try! @page_allocator.PageIdStorage::new(count),
    count: 0,
    cursor: 0,
    request_slot: -1,
    request_generation: 0,
    entry: Empty,
    spill_cursor: 0,
  })
}

///|
pub fn Scheduler::streaming_enabled(self : Scheduler) -> Bool {
  self.streaming is Some(_)
}

///|
pub fn Scheduler::streaming_busy(self : Scheduler) -> Bool {
  self.streaming is Some(owner) && (!(owner.flight is Empty) || owner.count > 0)
}

///|
fn StreamingOwner::command(
  self : StreamingOwner,
  opcode : @streaming_ir.Opcode,
  page : @page_allocator.PageId,
  host : @streaming_ir.HostPage,
) -> @streaming_ir.Command {
  {
    worker: self.worker,
    sequence: self.sequence,
    opcode,
    page: page.index(),
    page_generation: page.generation(),
    host,
  }
}

///|
fn StreamingOwner::drop(
  self : StreamingOwner,
  host : @streaming_ir.HostPage,
) -> Unit {
  if self.drop_count >= self.drops.length() {
    abort("streaming host accounting invariant")
  }
  self.drops[self.drop_count] = host
  self.drop_count += 1
}
