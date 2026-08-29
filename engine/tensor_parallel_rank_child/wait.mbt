///|
/// Blocks only after a nonblocking Configure read reported pending. The
/// inherited frame's absolute deadline remains authoritative.
pub fn TensorParallelRankChildConfigureReceiver::wait_pending(
  self : TensorParallelRankChildConfigureReceiver,
) -> Unit raise TensorParallelRankChildError {
  if !self.receive_started || self.configured {
    raise Invalid(InvalidLifecycle)
  }
  ignore(
    self.control.wait_receive(maximum_wait_millis=RANK_CHILD_IO_TIMEOUT_MILLIS) catch {
      error => raise Control(error)
    },
  )
}

///|
/// Paces collective startup with a short descriptor-readability poll, while
/// Ready publication waits on the actual pending response write.
pub fn TensorParallelRankChildStarting::wait_pending(
  self : TensorParallelRankChildStarting,
) -> Unit raise TensorParallelRankChildError {
  if self.ready_started {
    ignore(
      self.configured.receiver.control.wait_send(
        maximum_wait_millis=RANK_CHILD_IO_TIMEOUT_MILLIS,
      ) catch {
        error => raise Control(error)
      },
    )
  } else {
    ignore(
      self.configured.receiver.control.wait_peer_readable(
        maximum_wait_millis=RANK_CHILD_ACTIVE_WAIT_MILLIS,
      ) catch {
        error => raise Control(error)
      },
    )
  }
}

///|
/// Waits on the exact outstanding transport direction. An idle/completed rank
/// blocks for the parent command; an executing rank only performs a one-
/// millisecond readability wait so device and collective progress remains
/// nonblocking and independently advanced.
pub fn TensorParallelRankChildReady::wait_pending(
  self : TensorParallelRankChildReady,
) -> Unit raise TensorParallelRankChildError {
  if self.send_active {
    ignore(
      self.control_owner.wait_send(
        maximum_wait_millis=RANK_CHILD_IO_TIMEOUT_MILLIS,
      ) catch {
        error => raise Control(error)
      },
    )
    return
  }
  if !self.receive_active {
    raise Invalid(InvalidLifecycle)
  }
  let wait_millis = match self.execution {
    ChildExecuting(_, _) => RANK_CHILD_ACTIVE_WAIT_MILLIS
    _ => RANK_CHILD_IO_TIMEOUT_MILLIS
  }
  ignore(
    self.control_owner.wait_receive(maximum_wait_millis=wait_millis) catch {
      error => raise Control(error)
    },
  )
}
