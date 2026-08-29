///|
fn RankGroupProcessSupervisor::startup_write(
  self : RankGroupProcessSupervisor,
  rank : Int,
  length : Int,
) -> Bool {
  self.children[rank].startup_write(
    self.tx,
    length,
    self.limits.startup_io_timeout_millis,
  )
}

///|
fn RankGroupProcessSupervisor::startup_read(
  self : RankGroupProcessSupervisor,
  rank : Int,
) -> Int {
  self.children[rank].startup_read(
    self.rx,
    self.limits.startup_io_timeout_millis,
  )
}

///|
fn RankGroupProcessSupervisor::encode_configure(
  self : RankGroupProcessSupervisor,
  rank : Int,
) -> Int raise RankGroupProcessError {
  let startup = self.startups[rank]
  let payload_length = startup.configure.length()
  startup.configure.copy_to(self.tx, destination_offset=0) catch {
    error => self.fail(RankConfigure(error))
  }
  let length = @rank_group_wire.encode_rank_group_wire_payload_into(
    self.tx,
    self.wire_limits,
    startup.binding(),
    Configure,
    ParentToChild,
    None,
    0UL,
    self.tx,
    payload_length,
  ) catch {
    error => self.fail(Wire(error))
  }
  ignore(
    self.wire_transcript.accept_bytes(self.tx, length, self.wire_limits) catch {
      error => self.fail(Wire(error))
    },
  )
  length
}

///|
fn RankGroupProcessSupervisor::accept_ready(
  self : RankGroupProcessSupervisor,
  rank : Int,
  length : Int,
) -> Unit raise RankGroupProcessError {
  let header = self.wire_transcript.accept_bytes(
    self.rx,
    length,
    self.wire_limits,
  ) catch {
    error => self.fail(Wire(error))
  }
  let kind = header.kind()
  if kind != Ready {
    self.fail(StartupTransport(rank~))
  }
}

///|
fn RankGroupProcessSupervisor::spawn_children(
  self : RankGroupProcessSupervisor,
  authority : @approved_fs.WorkerApprovedRootSpawnAuthority,
) -> Bool {
  for rank in 0..<self.children.length() {
    match self.children[rank] {
      PreparedChild(prepared) => {
        let status = prepared.spawn_with_approved_roots_status(authority)
        let child = prepared.child_owner()
        self.children[rank] = LiveChild(child)
        if status != 0 {
          self.record_failure(Spawn(rank~))
          return false
        }
      }
      _ => {
        self.record_failure(Spawn(rank~))
        return false
      }
    }
  }
  true
}

///|
fn RankGroupProcessSupervisor::startup_handshake(
  self : RankGroupProcessSupervisor,
) -> Bool {
  // Configure every child before waiting for any Ready response.
  for rank in 0..<self.children.length() {
    self.io_rank = rank
    let length = try self.encode_configure(rank) catch {
      _ => return false
    } noraise {
      value => value
    }
    if !self.startup_write(rank, length) {
      self.record_failure(StartupTransport(rank~))
      return false
    }
  }
  for rank in 0..<self.children.length() {
    self.io_rank = rank
    let length = self.startup_read(rank)
    if length <= 0 {
      self.record_failure(StartupTransport(rank~))
      return false
    }
    self.accept_ready(rank, length) catch {
      _ => return false
    }
  }
  self.io_rank = -1
  true
}

///|

///|
/// Spawns every child through a capability-limited view of an externally
/// retained exact root pair. The view grants neither duplication nor close
/// authority. Cleanup deliberately leaves the pair to its sole outer owner,
/// which keeps it pinned through drain/reap and may use it for a fresh
/// whole-group replacement without acquiring a second persistent pair.
pub fn PreparedRankGroupProcess::spawn_with_approved_root_spawn_authority(
  self : PreparedRankGroupProcess,
  authority : @approved_fs.WorkerApprovedRootSpawnAuthority,
) -> RankGroupProcessPreparation raise RankGroupProcessError {
  let owner = self.0
  owner.require_lifecycle(Prepared)
  if owner.activation_consumed {
    owner.reject(Lifecycle)
  }
  owner.activation_consumed = true
  if !owner.spawn_children(authority) || !owner.startup_handshake() {
    return RankGroupProcessCleanupRequired(owner)
  }
  owner.lifecycle_value = Ready
  ReadyRankGroupProcess(owner)
}
