///|
/// Runs one independently admitted worker for exactly one authenticated group
/// generation. Uses the existing Submit/Poll/Complete/Abort/Drain/Close engine.
/// The caller retains the worker on error and must retry failed deterministic
/// cleanup; a transport failure is never reported as successful device release.
pub async fn serve(
  channel : @remote_tls.AuthenticatedRemoteChannel,
  worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner,
  limits : @rank_group_wire.RankGroupWireLimits,
) -> Unit {
  defer channel.close()
  errdefer (worker.remote_transport_lost() catch { _ => () })
  let (remote, _) = worker.remote_contract()
  let binding = FixedArray::make(
    @remote_startup.remote_startup_frame_bytes(),
    b'\x00',
  )
  ignore(remote.copy_to(binding))
  channel.verify_binding(Bytes::makei(binding.length(), i => binding[i]))
  let mailbox = @remote_mailbox.RemoteMailbox::new(limits.max_frame_bytes())
  let control = @rank_child_control.open_remote_rank_child_control(
    mailbox, limits,
  )
  @remote_session.with_session(
    channel,
    mailbox,
    Worker,
    generation=remote.group_generation_value(),
    lease_millis=remote.lease_millis(),
    heartbeat=sequence => {
      worker.renew_remote_lease(
        generation=remote.group_generation_value(),
        sequence~,
      )
    },
    () => {
      control.begin_receive()
      while !control.progress_receive().has_inbound() {
        worker.poll_remote_lease()
        @async.sleep(1)
      }
      let starting = @tensor_parallel_rank_child.start_remote_rank_child(
        control, worker,
      )
      let ready = for ;; {
        match starting.progress() {
          Some(ready) => break ready
          None => @async.sleep(1)
        }
      }
      for ;; {
        match ready.progress_control() {
          ControlClosed => break
          ControlPending => @async.sleep(1)
          ControlAdvanced => @async.sleep(0)
        }
      }
    },
  )
}
