///|
fn RankGroupChildOwner::startup_write(
  self : RankGroupChildOwner,
  source : FixedArray[Byte],
  length : Int,
  timeout_millis : Int,
) -> Bool {
  match self {
    LiveChild(child) =>
      child.startup_write_frame_status(source, length, timeout_millis) == 0
    PreparedChild(_) => false
  }
}

///|
fn RankGroupChildOwner::startup_read(
  self : RankGroupChildOwner,
  destination : FixedArray[Byte],
  timeout_millis : Int,
) -> Int {
  match self {
    LiveChild(child) =>
      child.startup_read_frame_status(destination, timeout_millis)
    PreparedChild(_) => -2
  }
}

///|
fn RankGroupChildOwner::begin_write(
  self : RankGroupChildOwner,
  source : FixedArray[Byte],
  length : Int,
  timeout_millis : Int,
) -> Unit raise RankGroupProcessError {
  match self {
    LiveChild(child) =>
      child.begin_write_frame(
        source,
        offset=0,
        byte_count=length,
        timeout_millis~,
      ) catch {
        _ => raise Failed(TransportLost(rank=-1))
      }
    PreparedChild(_) => raise Invalid(Lifecycle)
  }
}

///|
fn RankGroupChildOwner::begin_write_until(
  self : RankGroupChildOwner,
  source : FixedArray[Byte],
  length : Int,
  deadline_millis : Int64,
) -> Unit raise RankGroupProcessError {
  match self {
    LiveChild(child) =>
      child.begin_write_frame_until(
        source,
        offset=0,
        byte_count=length,
        deadline_millis~,
      ) catch {
        _ => raise Failed(TransportLost(rank=-1))
      }
    PreparedChild(_) => raise Invalid(Lifecycle)
  }
}

///|
fn RankGroupChildOwner::progress_write(
  self : RankGroupChildOwner,
) -> Bool raise RankGroupProcessError {
  match self {
    LiveChild(child) =>
      child.progress_write_frame() catch {
        _ => raise Failed(TransportLost(rank=-1))
      }
    PreparedChild(_) => raise Invalid(Lifecycle)
  }
}

///|
fn RankGroupChildOwner::begin_read(
  self : RankGroupChildOwner,
  destination : FixedArray[Byte],
  staging : FixedArray[Byte],
  timeout_millis : Int,
) -> Unit raise RankGroupProcessError {
  match self {
    LiveChild(child) =>
      child.begin_read_frame(destination, staging~, timeout_millis~) catch {
        _ => raise Failed(TransportLost(rank=-1))
      }
    PreparedChild(_) => raise Invalid(Lifecycle)
  }
}

///|
fn RankGroupChildOwner::begin_read_until(
  self : RankGroupChildOwner,
  destination : FixedArray[Byte],
  staging : FixedArray[Byte],
  deadline_millis : Int64,
) -> Unit raise RankGroupProcessError {
  match self {
    LiveChild(child) =>
      child.begin_read_frame_until(destination, staging~, deadline_millis~) catch {
        _ => raise Failed(TransportLost(rank=-1))
      }
    PreparedChild(_) => raise Invalid(Lifecycle)
  }
}

///|
fn RankGroupChildOwner::progress_read(
  self : RankGroupChildOwner,
) -> Int? raise RankGroupProcessError {
  match self {
    LiveChild(child) =>
      child.progress_read_frame() catch {
        _ => raise Failed(TransportLost(rank=-1))
      }
    PreparedChild(_) => raise Invalid(Lifecycle)
  }
}

///|
fn RankGroupChildOwner::observe_idle_exit(
  self : RankGroupChildOwner,
) -> @process.ChildExit? raise RankGroupProcessError {
  match self {
    LiveChild(child) => {
      let observed = child.observe_idle_exit() catch {
        _ => raise Failed(TransportLost(rank=-1))
      }
      if observed.has_exited() {
        Some(
          observed.exit() catch {
            _ => raise Failed(TransportLost(rank=-1))
          },
        )
      } else {
        None
      }
    }
    PreparedChild(_) => raise Invalid(Lifecycle)
  }
}

///|
fn RankGroupChildOwner::begin_cleanup(
  self : RankGroupChildOwner,
) -> Bool raise RankGroupProcessError {
  match self {
    PreparedChild(prepared) => prepared.child_owner().is_closed()
    LiveChild(child) => {
      if child.is_closed() {
        return true
      }
      child.begin_shutdown_maintenance() catch {
        _ => raise Invalid(Cleanup)
      }
      false
    }
  }
}

///|
fn RankGroupChildOwner::progress_cleanup(
  self : RankGroupChildOwner,
) -> RankGroupChildCleanupProgress raise RankGroupProcessError {
  match self {
    PreparedChild(prepared) =>
      if prepared.child_owner().is_closed() {
        ChildCleanupComplete
      } else {
        raise Invalid(Cleanup)
      }
    LiveChild(child) => {
      if child.is_closed() {
        return ChildCleanupComplete
      }
      let progress = child.progress_shutdown_maintenance() catch {
        _ => raise Invalid(Cleanup)
      }
      if progress.is_complete() {
        ChildCleanupComplete
      } else if progress.is_cleanup_stuck() {
        ChildCleanupStuck
      } else if progress.is_cleanup_required() {
        raise Invalid(Cleanup)
      } else if progress.is_advanced() {
        ChildCleanupAdvanced
      } else {
        ChildCleanupPending
      }
    }
  }
}
