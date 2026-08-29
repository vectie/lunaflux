///|
fn RankGroupProcessSupervisor::mark_survivor_lost(
  self : RankGroupProcessSupervisor,
  rank : Int,
) -> Unit {
  if rank < 0 || rank >= self.children.length() || self.cleanup_lost[rank] {
    return
  }
  self.cleanup_lost[rank] = true
  if self.protocol.lifecycle() == Faulted {
    self.protocol.mark_drain_rank_lost(rank) catch {
      _ => ()
    }
  }
}

///|
fn RankGroupProcessSupervisor::next_survivor_rank(
  self : RankGroupProcessSupervisor,
  start : Int,
) -> Int {
  for rank in start..<self.children.length() {
    if !self.cleanup_lost[rank] {
      return rank
    }
  }
  -1
}

///|
fn RankGroupProcessSupervisor::finish_survivor_phase(
  self : RankGroupProcessSupervisor,
) -> Unit raise RankGroupProcessError {
  if self.survivor_phase == 0 {
    self.protocol.close_faulted() catch {
      error => raise Failed(Protocol(error))
    }
    self.survivor_phase = 1
    self.survivor_rank = -1
    self.stage_next_survivor(0)
  } else {
    self.survivor_terminal = false
    self.survivor_rank = -1
    self.io_rank = -1
    self.io_phase = 0
    self.begin_child_cleanup()
  }
}

///|
fn RankGroupProcessSupervisor::stage_next_survivor(
  self : RankGroupProcessSupervisor,
  start : Int,
) -> Unit raise RankGroupProcessError {
  let mut next = self.next_survivor_rank(start)
  while next >= 0 {
    let kind = if self.survivor_phase == 0 {
      @rank_group_wire.Drain
    } else {
      Close
    }
    let length = @rank_group_wire.encode_rank_group_wire_control_into(
      self.tx,
      self.wire_limits,
      self.binding(next),
      kind,
      ParentToChild,
      None,
      0UL,
    ) catch {
      error => raise Failed(Wire(error))
    }
    try
      self.children[next].begin_write(
        self.tx,
        length,
        self.limits.io_timeout_millis,
      )
    catch {
      _ => {
        self.mark_survivor_lost(next)
        next = self.next_survivor_rank(next + 1)
        continue
      }
    } noraise {
      _ => ()
    }
    self.survivor_rank = next
    self.io_rank = next
    self.io_phase = 1
    return
  }
  self.survivor_rank = -2
  self.io_rank = -1
  self.io_phase = 0
  self.finish_survivor_phase()
}

///|
fn RankGroupProcessSupervisor::begin_survivor_cleanup(
  self : RankGroupProcessSupervisor,
) -> Unit raise RankGroupProcessError {
  if self.survivor_terminal {
    if self.io_phase != 0 {
      return
    }
    if self.survivor_rank == -2 {
      self.finish_survivor_phase()
    } else if self.survivor_rank == -1 {
      self.stage_next_survivor(0)
    } else {
      self.stage_next_survivor(self.survivor_rank + 1)
    }
    return
  }
  self.survivor_terminal = true
  self.survivor_phase = 0
  self.survivor_rank = -1
  self.io_rank = -1
  self.io_phase = 0
  self.stage_next_survivor(0)
}

///|
fn RankGroupProcessSupervisor::accept_survivor_response(
  self : RankGroupProcessSupervisor,
  rank : Int,
  length : Int,
) -> Unit raise RankGroupProcessError {
  let header = @rank_group_wire.validate_rank_group_wire_binding(
    self.rx,
    length,
    self.wire_limits,
    self.binding(rank),
  ) catch {
    _ => {
      self.mark_survivor_lost(rank)
      self.stage_next_survivor(rank + 1)
      return
    }
  }
  let expected = if self.survivor_phase == 0 {
    @rank_group_wire.Drained
  } else {
    Closed
  }
  if header.direction() != ChildToParent ||
    header.kind() != expected ||
    header.slot() is Some(_) ||
    header.sequence_value() != 0UL ||
    header.payload_length() != 0 {
    self.mark_survivor_lost(rank)
    self.stage_next_survivor(rank + 1)
    return
  }
  if self.survivor_phase == 0 {
    let lease = self.protocol.drain_lease(rank) catch {
      error => raise Failed(Protocol(error))
    }
    let ack = lease.ack() catch { error => raise Failed(Protocol(error)) }
    self.protocol.accept_drain_ack(ack) catch {
      error => raise Failed(Protocol(error))
    }
  }
  self.stage_next_survivor(rank + 1)
}

///|
fn RankGroupProcessSupervisor::progress_survivor_cleanup(
  self : RankGroupProcessSupervisor,
) -> RankGroupProcessCleanupProgress raise RankGroupProcessError {
  if !self.survivor_terminal || self.lifecycle_value != Faulted {
    self.reject(Lifecycle)
  }
  let rank = self.survivor_rank
  if rank == -2 {
    self.finish_survivor_phase()
    return CleanupAdvanced
  }
  if rank == -1 && self.io_phase == 0 {
    self.stage_next_survivor(0)
    return CleanupAdvanced
  }
  if rank < 0 || self.io_phase == 0 {
    self.reject(Lifecycle)
  }
  if self.io_phase == 1 {
    let written = try self.children[rank].progress_write() catch {
      _ => {
        self.mark_survivor_lost(rank)
        self.stage_next_survivor(rank + 1)
        return CleanupAdvanced
      }
    } noraise {
      value => value
    }
    if !written {
      return CleanupPending
    }
    self.children[rank].begin_read(
      self.rx,
      self.rx_staging,
      self.limits.io_timeout_millis,
    ) catch {
      _ => {
        self.mark_survivor_lost(rank)
        self.stage_next_survivor(rank + 1)
        return CleanupAdvanced
      }
    }
    self.io_phase = 2
    return CleanupAdvanced
  }
  let received = try self.children[rank].progress_read() catch {
    _ => {
      self.mark_survivor_lost(rank)
      self.stage_next_survivor(rank + 1)
      return CleanupAdvanced
    }
  } noraise {
    value => value
  }
  match received {
    None => CleanupPending
    Some(length) => {
      self.io_phase = 0
      self.accept_survivor_response(rank, length)
      CleanupAdvanced
    }
  }
}
