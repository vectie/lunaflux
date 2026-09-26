///|
/// The result always owns cleanup, even after native construction fails.
pub fn Cache::prepare(
  context : @device.Context,
  allocation : @device.Allocation,
  plan : Plan,
) -> Cache raise @device.DeviceError {
  if allocation.byte_count() != plan.layout.arena_bytes() {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  let transfers = context.prepare_transfer_pool(
    allocation,
    host_bytes=plan.host_bytes(),
    lanes=plan.transfer_lanes(),
    max_regions_per_lane=plan.components,
  )
  {
    plan,
    residency: @residency.Pool::new(plan.limits),
    transfers,
    lanes: FixedArray::make(plan.transfer_lanes(), Idle),
    poisoned: !transfers.is_prepared(),
    closed: false,
  }
}

///|
fn Cache::free_lane(self : Cache) -> Int {
  for i in 0..<self.lanes.length() {
    if self.lanes[i] is Idle {
      return i
    }
  }
  -1
}

///|
fn Cache::fault(
  self : Cache,
  lane : Int,
  transfer : @residency.Transfer,
) -> Unit {
  let _ = self.residency.fail(transfer)
  self.lanes[lane] = Fault(transfer)
  self.poisoned = true
}

///|
fn Cache::submit(
  self : Cache,
  lane : Int,
  transfer : @residency.Transfer,
  page : Int,
  to_device : Bool,
) -> Start raise @device.DeviceError {
  self.lanes[lane] = Active(transfer)
  errdefer self.fault(lane, transfer)
  for component in 0..<self.plan.components {
    let region = self.plan.region(transfer.slot(), page, component)
    self.transfers.enqueue(
      lane~,
      host_offset=region.host_offset,
      device_offset=region.device_offset,
      bytes=region.bytes,
      to_device~,
    )
  }
  self.transfers.record(lane)
  Started({ owner: self, lane, transfer, })
}

///|
pub fn Cache::begin_spill(
  self : Cache,
  page : Int,
) -> Start raise @device.DeviceError {
  if !self.is_prepared() || page < 0 || page >= self.plan.pages {
    return Rejected
  }
  let lane = self.free_lane()
  if lane < 0 {
    return Backpressured
  }
  match self.residency.begin_spill() {
    Started(transfer) => self.submit(lane, transfer, page, false)
    Backpressured => Backpressured
    _ => Rejected
  }
}

///|
pub fn Cache::begin_restore(
  self : Cache,
  entry : @residency.Entry,
  page : Int,
) -> Start raise @device.DeviceError {
  if !self.is_prepared() || page < 0 || page >= self.plan.pages {
    return Rejected
  }
  let lane = self.free_lane()
  if lane < 0 {
    return Backpressured
  }
  match self.residency.begin_restore(entry) {
    Started(transfer) => self.submit(lane, transfer, page, true)
    Backpressured => Backpressured
    _ => Rejected
  }
}

///|
fn Cache::owns(self : Cache, operation : Operation) -> Bool {
  if !physical_equal(self, operation.owner) || self.closed {
    return false
  }
  match self.lanes[operation.lane] {
    Active(transfer) => transfer.same_attempt(operation.transfer)
    _ => false
  }
}

///|
pub fn Cache::cancel(self : Cache, operation : Operation) -> Bool {
  self.owns(operation) &&
  self.residency.cancel(operation.transfer) is RetainUntilCompletion
}

///|
pub fn Cache::poll(
  self : Cache,
  operation : Operation,
) -> Progress raise @device.DeviceError {
  if !self.owns(operation) {
    return Stale
  }
  errdefer self.fault(operation.lane, operation.transfer)
  let complete = self.transfers.poll(operation.lane)
  if !complete {
    return Pending
  }
  self.transfers.reset(operation.lane)
  let effect = self.residency.complete(operation.transfer)
  self.lanes[operation.lane] = Idle
  match effect {
    HostReady => HostReady
    DeviceReady => DeviceReady
    Released | RestoreCancelled => Cancelled
    _ => {
      self.poisoned = true
      Stale
    }
  }
}

///|
pub fn Cache::evict(self : Cache, entry : @residency.Entry) -> Bool {
  self.is_prepared() && self.residency.evict(entry) is Released
}

///|
/// Fault/terminal path. A failed drain retains every lane and allocation lease.
pub fn Cache::close(self : Cache) -> Unit raise @device.DeviceError {
  if self.closed {
    return
  }
  self.poisoned = true
  for lane in 0..<self.lanes.length() {
    match self.lanes[lane] {
      Idle => ()
      Active(transfer) | Fault(transfer) => {
        let _ = self.residency.fail(transfer)
        self.lanes[lane] = Fault(transfer)
        self.transfers.drain(lane)
        self.transfers.reset(lane)
        let _ = self.residency.drained(transfer)
        self.lanes[lane] = Idle
      }
    }
  }
  self.transfers.close()
  let _ = self.residency.close()
  self.closed = true
}
