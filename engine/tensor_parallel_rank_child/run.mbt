///|
const RANK_CHILD_MAX_FRAME_BYTES : Int = 67108864

///|
// The parent owns the shorter operational deadlines and kill/reap authority.
// The child admits only the process boundary's canonical maximum so it cannot
// reject a valid parent policy because of a hidden, shorter local timeout.
const RANK_CHILD_IO_TIMEOUT_MILLIS : Int = 600000

///|
const RANK_CHILD_ACTIVE_WAIT_MILLIS : Int = 1

///|
fn receive_rank_child_configure(
  receiver : TensorParallelRankChildConfigureReceiver,
) -> TensorParallelRankChildConfigured raise TensorParallelRankChildError {
  receiver.begin()
  for ;; {
    match receiver.progress() {
      ConfigurePending => receiver.wait_pending()
      ConfigureReady(configured) => return configured
    }
  }
}

///|
fn fail_rank_child_admission(
  configured : TensorParallelRankChildConfigured,
) -> Unit {
  configured.receiver.control.fail_local_admission() catch {
    _ => ()
  }
}

///|
fn close_failed_rank_child_startup(
  starting : TensorParallelRankChildStarting,
) -> Unit {
  starting.close_worker_after_failure() catch {
    _ => ()
  }
}

///|
fn close_failed_rank_child_control(
  ready : TensorParallelRankChildReady,
) -> Unit {
  ready.close_worker_after_failure() catch {
    _ => ()
  }
}

///|
fn finish_failed_rank_child_preparation(
  failed : FailedTensorParallelRankChildPreparation,
) -> Unit {
  failed.retry_close() catch {
    _ => ()
  }
}

///|
fn publish_rank_child_ready(
  starting : TensorParallelRankChildStarting,
) -> TensorParallelRankChildReady raise TensorParallelRankChildError {
  for ;; {
    match starting.progress_ready() {
      CollectivePending | ReadySending => starting.wait_pending()
      ReadyPublished => return starting.ready()
    }
  }
}

///|
fn serve_rank_child_control(ready : TensorParallelRankChildReady) -> Bool {
  for ;; {
    let progress = ready.progress_control() catch {
      _ => {
        close_failed_rank_child_control(ready)
        return false
      }
    }
    match progress {
      ControlPending =>
        ready.wait_pending() catch {
          _ => {
            close_failed_rank_child_control(ready)
            return false
          }
        }
      ControlAdvanced => ()
      ControlClosed => return true
    }
  }
}

///|
/// Runs the real tensor-parallel rank child over the inherited private channel.
/// Every individual read, collective, execution, and write progress call is
/// bounded. `false` is payload-free failure evidence for the executable's
/// silent nonzero-exit boundary.
pub fn run() -> Bool {
  let wire_limits = @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=RANK_CHILD_MAX_FRAME_BYTES -
      @rank_group_wire.rank_group_wire_header_bytes(),
  ) catch {
    _ => return false
  }
  let receiver = open_configure_receiver(
    wire_limits,
    io_timeout_millis=RANK_CHILD_IO_TIMEOUT_MILLIS,
    shutdown_timeout_millis=RANK_CHILD_IO_TIMEOUT_MILLIS,
  ) catch {
    _ => return false
  }
  let configured = receive_rank_child_configure(receiver) catch {
    _ => return false
  }
  let preparation = prepare_configured_rank_child(configured) catch {
    _ => {
      fail_rank_child_admission(configured)
      return false
    }
  }
  match preparation {
    RankChildCleanupRequired(failed) => {
      finish_failed_rank_child_preparation(failed)
      fail_rank_child_admission(configured)
      false
    }
    RankChildStarting(starting) => {
      let ready = publish_rank_child_ready(starting) catch {
        _ => {
          close_failed_rank_child_startup(starting)
          return false
        }
      }
      serve_rank_child_control(ready)
    }
  }
}
