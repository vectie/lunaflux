///|
/// Pure rank-result admission. A group publishes only terminal results from
/// every shard; cancellation may race with a completed restore on any rank.
pub fn rank_status_valid(
  original : Opcode,
  status : Status,
  cancelled : Bool,
) -> Bool {
  match (original, status) {
    (Spill | Restore, Pending) => true
    (Spill, HostReady) | (Restore, DeviceReady) | (Evict, Evicted) => true
    (Restore, Cancelled) => cancelled
    _ => false
  }
}

///|
pub fn group_status(
  original : Opcode,
  remaining : Int,
  cancelled : Bool,
) -> Status {
  if remaining != 0 {
    return Pending
  }
  match original {
    Spill => HostReady
    Restore => if cancelled { Cancelled } else { DeviceReady }
    Evict => Evicted
    _ => Rejected
  }
}

///|
/// Divides the group host budget before admitting a complete rank-local shard.
pub fn Capability::for_rank(
  self : Capability,
  page_bytes : Int64,
  world_size : Int,
) -> Capability raise StreamingError {
  if world_size < 2 || world_size > 16 {
    raise InvalidCapability
  }
  Capability::new(
    slots=self.slots,
    host_bytes=self.host_bytes / world_size.to_int64(),
    page_bytes~,
    page_count=self.page_count,
    reserve_pages=self.reserve_pages,
    restore_ns=self.restore_ns,
    recompute_ns=self.recompute_ns,
    separate_host_memory=true,
    transfer_timeout_millis=self.transfer_timeout_millis,
    workspace_budget=self.workspace_budget,
    shapes=self.shapes[:],
  )
}

///|
/// Immutable all-rank publication barrier. The mask makes duplicate results
/// and skipped terminal ranks explicit without heap allocation.
#valtype
pub struct GroupTransfer {
  command : Command
  ranks : Int
  finished : UInt
  remaining : Int
  host : HostPage
  cancelled : Bool
}

///|
pub fn GroupTransfer::new(
  command : Command,
  ranks : Int,
) -> GroupTransfer raise StreamingError {
  if ranks < 2 || ranks > 16 {
    raise InvalidCapability
  }
  {
    command,
    ranks,
    finished: 0U,
    remaining: ranks,
    host: command.host,
    cancelled: false,
  }
}

///|
pub fn GroupTransfer::rank_done(self : GroupTransfer, rank : Int) -> Bool {
  rank >= 0 && rank < self.ranks && (self.finished & (1U << rank)) != 0U
}

///|
pub fn GroupTransfer::cancel(
  self : GroupTransfer,
) -> GroupTransfer raise StreamingError {
  if !(self.command.opcode is Restore) {
    raise InvalidTransition
  }
  { ..self, cancelled: true, }
}

///|
pub fn GroupTransfer::accept(
  self : GroupTransfer,
  rank : Int,
  reply : Reply,
) -> GroupTransfer raise StreamingError {
  if rank < 0 ||
    rank >= self.ranks ||
    self.rank_done(rank) ||
    !reply.command.same_transaction(self.command) ||
    !rank_status_valid(self.command.opcode, reply.status, self.cancelled) {
    raise InvalidTransition
  }
  if reply.status is Pending {
    return self
  }
  let host = if self.command.opcode is Spill && self.remaining == self.ranks {
    reply.host
  } else {
    self.host
  }
  if host.slot < 0 || host.generation <= 0 || !host.same(reply.host) {
    raise InvalidTransition
  }
  {
    ..self,
    finished: self.finished | (1U << rank),
    remaining: self.remaining - 1,
    host,
  }
}

///|
pub fn GroupTransfer::reply(self : GroupTransfer, command : Command) -> Reply {
  {
    command,
    status: group_status(self.command.opcode, self.remaining, self.cancelled),
    host: self.host,
  }
}
