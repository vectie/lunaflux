///|
test "remote watchdog polls repeatedly without consuming the clock capability" {
  let clock = @monotonic_clock.MonotonicClock::system().prepare_poller()
  assert_true(clock.poll())
  let watchdog : RemoteWorkerWatchdog = {
    clock,
    lease: @remote_lease.RemoteLease::start(
      generation=1UL,
      duration_millis=300000UL,
      local_now_millis=clock.value(),
    ),
  }
  for _ in 0..<1000 {
    poll_watchdog(watchdog)
    assert_eq(watchdog.lease.status(), @remote_lease.Active)
  }
  watchdog.lease = watchdog.lease.transport_lost()
  poll_watchdog(watchdog)
  assert_eq(watchdog.lease.status(), @remote_lease.TransportLost)
}
