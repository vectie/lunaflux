///|
// Emulates an already released rank over the real authenticated receipt wire.
// Native resource release itself remains a physical worker qualification gate.
async fn remote_test_release(
  startup : @remote_startup.RemoteRankStartup,
) -> @remote_release.RemoteReleaseReceipt {
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  @async.with_task_group() <| tasks => {
    tasks.spawn_bg(() => {
      let (socket, _) = listener.accept()
      let channel = @remote_tls.accept(
        socket,
        @remote_tls.RemoteCredential::from_bytes(Bytes::make(32, b'x')),
        remote_test_binding(startup),
        certificate_pem_file="runtime/remote_tls/test_keys/cert.pem",
        private_key_pem_file="runtime/remote_tls/test_keys/key.pem",
        max_frame_bytes=16,
        timeout_millis=1000,
      )
      defer channel.close()
      let bytes = FixedArray::make(16, b'\x00')
      assert_eq(channel.receive(bytes), 16)
      assert_eq(Bytes::makei(16, i => bytes[i]), b"LFRELEASEQUERY01")
      for i in 0..<16 {
        bytes[i] = b"LFRELEASEDONE001"[i]
      }
      channel.send(bytes, 16)
      assert_eq(channel.receive(bytes), 16)
      assert_eq(Bytes::makei(16, i => bytes[i]), b"LFRELEASEACK0001")
    })
    let channel = @remote_tls.connect(
      @socket.Tcp::connect(listener.addr),
      @remote_tls.RemoteCredential::from_bytes(Bytes::make(32, b'x')),
      remote_test_binding(startup),
      server_name="localhost",
      ca_pem_file="runtime/remote_tls/test_keys/cert.pem",
      max_frame_bytes=16,
      timeout_millis=1000,
    )
    @remote_release.receive(channel, startup)
  }
}

///|
async fn exercise_remote_recovery(
  model : @plan.ModelPlan,
  initial : @remote_startup.RemoteGroupStartup,
  admitted : TensorParallelGroupBootstrap,
  replacement : @remote_startup.RemoteGroupStartup,
  replacement_admitted : TensorParallelGroupBootstrap,
  final_group : @remote_startup.RemoteGroupStartup,
  final_admitted : TensorParallelGroupBootstrap,
) -> Unit {
  let pending = remote_test_service_preparation(initial, admitted, model)
  let control = pending.control()
  let group = control.group()
  let limits = @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=16384,
  )
  let ranks = [0, 1].map(_ => {
    ServiceRankEmulator::new(initial.rank(0).unwrap().worker(), limits)
  })
  let service = for _ in 0..<100 {
    for rank in 0..<2 {
      ranks[rank].progress(group, rank)
    }
    if pending.poll_ready() is Some(ready) {
      break ready.take_raw_ready()
    }
  } nobreak {
    fail("initial remote service did not start")
  }
  group.transport_lost()
  service.begin_recovery_maintenance(ExecutionFailed)
  assert_true(
    service.progress_maintenance() ==
    @worker_service.LunaWorkerServiceMaintenanceCleanupStuck,
  )
  let old_receipt = remote_test_release(initial.rank(0).unwrap())
  control.confirm_release(0, old_receipt)
  assert_false(group.recovery_cleanup_complete())
  control.confirm_release(1, remote_test_release(initial.rank(1).unwrap()))
  for _ in 0..<10 {
    ignore(service.progress_maintenance())
    if service.lifecycle() == @worker_service.ServiceRestartReady {
      break
    }
  }
  assert_true(service.lifecycle() == @worker_service.ServiceRestartReady)
  let failed = control.prepare_replacement(replacement, replacement_admitted)
  failed.mailbox(1).fail()
  try failed.poll_startup() catch {
    _ => ()
  } noraise {
    _ => fail("failed replacement became ready")
  }
  assert_true(service.restart() is @worker_service.RestartCleanupRequired)
  service.begin_restart_cleanup_maintenance()
  try control.confirm_replacement_release(0, old_receipt) catch {
    _ => ()
  } noraise {
    _ => fail("old receipt released a replacement")
  }
  assert_true(
    service.progress_maintenance() ==
    @worker_service.LunaWorkerServiceMaintenanceCleanupStuck,
  )
  for rank in 0..<2 {
    control.confirm_replacement_release(
      rank,
      remote_test_release(replacement.rank(rank).unwrap()),
    )
  }
  ignore(service.progress_maintenance())
  assert_true(service.lifecycle() == @worker_service.ServiceRestartReady)
  try control.prepare_replacement(replacement, replacement_admitted) catch {
    _ => ()
  } noraise {
    _ => fail("failed generation reused")
  }
  let next = control.prepare_replacement(final_group, final_admitted)
  let next_ranks = [0, 1].map(_ => {
    ServiceRankEmulator::new(final_group.rank(0).unwrap().worker(), limits)
  })
  for _ in 0..<100 {
    for rank in 0..<2 {
      next_ranks[rank].progress(next, rank)
    }
    if next.poll_startup() {
      break
    }
  }
  assert_true(service.restart() is @worker_service.Restarted)
  service.begin_shutdown_maintenance()
  for _ in 0..<100 {
    for rank in 0..<2 {
      next_ranks[rank].progress(next, rank)
    }
    ignore(service.progress_maintenance())
    if service.lifecycle() == @worker_service.ServiceClosed {
      break
    }
  }
  assert_true(service.lifecycle() == @worker_service.ServiceClosed)
}
