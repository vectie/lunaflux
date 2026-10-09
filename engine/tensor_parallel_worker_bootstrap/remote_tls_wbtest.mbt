///|
fn remote_test_binding(
  startup : @remote_startup.RemoteRankStartup,
) -> Bytes raise {
  let bytes = FixedArray::make(
    @remote_startup.remote_startup_frame_bytes(),
    b'\x00',
  )
  ignore(startup.copy_to(bytes))
  Bytes::makei(bytes.length(), i => bytes[i])
}

///|
async fn mock_remote_worker(
  listener : @socket.TcpServer,
  startup : @remote_startup.RemoteRankStartup,
  limits : @rank_group_wire.RankGroupWireLimits,
) -> Unit {
  let (socket, _) = listener.accept()
  let channel = @remote_tls.accept(
    socket,
    @remote_tls.RemoteCredential::from_bytes(Bytes::make(32, b'x')),
    remote_test_binding(startup),
    certificate_pem_file="runtime/remote_tls/test_keys/cert.pem",
    private_key_pem_file="runtime/remote_tls/test_keys/key.pem",
    max_frame_bytes=limits.max_frame_bytes() + 1,
    timeout_millis=1000,
  )
  let mailbox = @remote_mailbox.RemoteMailbox::new(limits.max_frame_bytes())
  let control = @rank_child_control.open_remote_rank_child_control(
    mailbox, limits,
  )
  @remote_session.with_session(
    channel,
    mailbox,
    Worker,
    generation=startup.group_generation_value(),
    lease_millis=startup.lease_millis(),
    heartbeat=_ => (),
    () => {
      for step in 0..<3 {
        control.begin_receive()
        while !control.progress_receive().has_inbound() {
          @async.sleep(1)
        }
        let inbound = control.inbound()
        match step {
          0 =>
            control.begin_ready(
              inbound,
              inbound.binding(),
              previous_sequence_value=0UL,
            )
          1 => control.begin_drained(inbound)
          _ => control.begin_closed(inbound)
        }
        while !control.progress_send().is_complete() {
          @async.sleep(1)
        }
      }
    },
  )
}

///|
async fn exercise_remote_tls_group(
  remote : @remote_startup.RemoteGroupStartup,
  admitted : TensorParallelGroupBootstrap,
) -> Unit {
  let limits = @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=16384,
  )
  let worker = remote.rank(0).unwrap().worker()
  let first = @worker_protocol.SchedulePlanBuffer::new(worker.worker_limits())
  let second = @worker_protocol.SchedulePlanBuffer::new(worker.worker_limits())
  let group = @remote_group.RemoteGroup::new(
    remote,
    admitted,
    [first.identity(), second.identity()],
    limits,
    timeout_millis=5000UL,
  )
  let zero = @socket.TcpServer::TcpServer(@socket.Addr::parse("127.0.0.1:0"))
  defer zero.close()
  let one = @socket.TcpServer::TcpServer(@socket.Addr::parse("127.0.0.1:0"))
  defer one.close()
  @async.with_task_group() <| tasks => {
    tasks.spawn_bg(() => {
      mock_remote_worker(zero, remote.rank(0).unwrap(), limits)
    })
    tasks.spawn_bg(() => {
      mock_remote_worker(one, remote.rank(1).unwrap(), limits)
    })
    let rank_zero = @remote_tls.connect(
      @socket.Tcp::connect(zero.addr),
      @remote_tls.RemoteCredential::from_bytes(Bytes::make(32, b'x')),
      remote_test_binding(remote.rank(0).unwrap()),
      server_name="localhost",
      ca_pem_file="runtime/remote_tls/test_keys/cert.pem",
      max_frame_bytes=limits.max_frame_bytes() + 1,
      timeout_millis=1000,
    )
    defer rank_zero.close()
    let rank_one = @remote_tls.connect(
      @socket.Tcp::connect(one.addr),
      @remote_tls.RemoteCredential::from_bytes(Bytes::make(32, b'x')),
      remote_test_binding(remote.rank(1).unwrap()),
      server_name="localhost",
      ca_pem_file="runtime/remote_tls/test_keys/cert.pem",
      max_frame_bytes=limits.max_frame_bytes() + 1,
      timeout_millis=1000,
    )
    @remote_coordinator.with_group(group, rank_zero, rank_one, () => {
      while !group.poll_startup() {
        @async.sleep(1)
      }
      @async.sleep(1100)
      group.probe_idle_health()
      group.begin_close()
      while !group.progress_close() {
        @async.sleep(1)
      }
    })
  }
  assert_true(group.lifecycle() == @remote_group.Closed)
}
