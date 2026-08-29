///|
/// Performs one round-robin WNOHANG observation only while the group is idle.
pub fn RankGroupProcessSupervisor::probe_idle_rank(
  self : RankGroupProcessSupervisor,
) -> Unit raise RankGroupProcessError {
  self.require_lifecycle(Ready)
  let rank = self.next_rank
  self.next_rank = (rank + 1) % self.children.length()
  let exited = self.children[rank].observe_idle_exit() catch {
    _ => self.fail(TransportLost(rank~))
  }
  match exited {
    Some(value) => {
      self.cleanup_started[rank] = true
      self.protocol.report_idle_rank_loss(rank) catch {
        error => self.fail(Protocol(error))
      }
      self.fail(IdleRankExit(rank~, kind=value.kind(), code=value.code()))
    }
    None => ()
  }
}

///|
fn RankGroupProcessSupervisor::begin_child_cleanup(
  self : RankGroupProcessSupervisor,
) -> Unit {
  for rank in 0..<self.children.length() {
    if !self.cleanup_started[rank] {
      let started = self.children[rank].begin_cleanup() catch {
        _ => {
          self.record_cleanup_failure(RankChildCleanupFailed(rank~))
          continue
        }
      }
      ignore(started)
      self.cleanup_started[rank] = true
    }
  }
  self.cleanup_cursor = 0
  self.lifecycle_value = Cleaning
}

///|
/// Revokes pending transfers and starts whole-group terminate/reap.
pub fn RankGroupProcessSupervisor::begin_cleanup(
  self : RankGroupProcessSupervisor,
) -> Unit raise RankGroupProcessError {
  self.require_lifecycle(Faulted)
  if self.protocol.lifecycle() == Closed {
    self.begin_child_cleanup()
  } else if self.protocol.lifecycle() == Ready {
    // A local pre-submission failure has no device plan to abort or protocol
    // drain leases to settle. Close the inert owner, then reap every child.
    self.protocol.close() catch {
      error => self.fail(Protocol(error))
    }
    self.begin_child_cleanup()
  } else {
    self.begin_survivor_cleanup()
  }
}

///|
/// Starts the canonical healthy Ready -> Drain/Drained -> Close/Closed wire
/// sequence. OS child cleanup cannot begin until every rank has acknowledged
/// explicit runtime release.
pub fn RankGroupProcessSupervisor::begin_close(
  self : RankGroupProcessSupervisor,
) -> Unit raise RankGroupProcessError {
  self.require_lifecycle(Ready)
  self.healthy_terminal = true
  self.lifecycle_value = Draining
  self.stage_terminal(0, Drain)
}

///|
fn RankGroupProcessSupervisor::child_closed(
  self : RankGroupProcessSupervisor,
  rank : Int,
) -> Bool {
  match self.children[rank] {
    PreparedChild(prepared) => prepared.child_owner().is_closed()
    LiveChild(owner) => owner.is_closed()
  }
}

///|
fn RankGroupProcessSupervisor::all_children_closed(
  self : RankGroupProcessSupervisor,
) -> Bool {
  for rank in 0..<self.children.length() {
    if !self.child_closed(rank) {
      return false
    }
  }
  true
}

///|
/// Advances at most one child maintenance transition globally.
pub fn RankGroupProcessSupervisor::progress_cleanup(
  self : RankGroupProcessSupervisor,
) -> RankGroupProcessCleanupProgress raise RankGroupProcessError {
  if self.lifecycle_value == Faulted && self.survivor_terminal {
    return self.progress_survivor_cleanup()
  }
  self.require_lifecycle(Cleaning)
  if self.all_children_closed() {
    self.lifecycle_value = Closed
    return CleanupComplete
  }
  let rank = self.cleanup_cursor
  self.cleanup_cursor = (rank + 1) % self.children.length()
  if self.child_closed(rank) {
    return CleanupAdvanced
  }
  if !self.cleanup_started[rank] {
    ignore(
      self.children[rank].begin_cleanup() catch {
        _ => {
          self.record_cleanup_failure(RankChildCleanupFailed(rank~))
          return CleanupPending
        }
      },
    )
    self.cleanup_started[rank] = true
    return CleanupAdvanced
  }
  let progress = self.children[rank].progress_cleanup() catch {
    _ => {
      self.record_cleanup_failure(RankChildCleanupFailed(rank~))
      return CleanupPending
    }
  }
  match progress {
    ChildCleanupPending => CleanupPending
    ChildCleanupAdvanced | ChildCleanupComplete => CleanupAdvanced
    ChildCleanupStuck => CleanupStuck(rank~)
  }
}
