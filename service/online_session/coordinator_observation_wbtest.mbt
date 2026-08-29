///|
fn observation_fixture_stream(
  compatibility : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedStream raise {
  let service = compatibility.service
  retire_fixture_stream(compatibility.stream)
  let opened = service.open_stream()
  service.take_open_stream(opened)
}

///|
priv struct LunaOnlineLatencyTestServer {
  mut observation_recorded : Bool
  mut first_token_samples : Int
  mut inter_token_samples : Int
  mut request_samples : Int
  mut first_token_millis : Int
  mut inter_token_millis : Int
  mut request_millis : Int
}

///|
fn LunaOnlineLatencyTestServer::record_before_ack(
  self : LunaOnlineLatencyTestServer,
  observation : LunaOnlineFramedObservation,
) -> Unit raise {
  if self.observation_recorded {
    return
  }
  let kind = observation.latency_kind()
  let millis = if kind == LunaOnlineFramedNoLatency {
    0
  } else {
    observation.latency_millis()
  }
  match kind {
    LunaOnlineFramedNoLatency => ()
    LunaOnlineFramedFirstTokenLatency => {
      self.first_token_samples += 1
      self.first_token_millis = millis
    }
    LunaOnlineFramedInterTokenLatency => {
      self.inter_token_samples += 1
      self.inter_token_millis = millis
    }
    LunaOnlineFramedRequestLatency => {
      self.request_samples += 1
      self.request_millis = millis
    }
  }
  self.observation_recorded = true
}

///|
fn LunaOnlineLatencyTestServer::ack(
  self : LunaOnlineLatencyTestServer,
  observation : LunaOnlineFramedObservation,
) -> Unit raise {
  observation.ack()
  self.observation_recorded = false
}

///|
fn LunaOnlineLatencyTestServer::record_retry_and_ack(
  self : LunaOnlineLatencyTestServer,
  observation : LunaOnlineFramedObservation,
) -> Unit raise {
  self.record_before_ack(observation)
  // Models the server re-entering the record path after an invariant ACK
  // failure: the retained bit suppresses a second metrics mutation.
  self.record_before_ack(observation)
  self.ack(observation)
}

///|
test "Luna admitted event sequence records latency once across ACK retry" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let stream = observation_fixture_stream(compatibility)
  let server : LunaOnlineLatencyTestServer = {
    observation_recorded: false,
    first_token_samples: 0,
    inter_token_samples: 0,
    request_samples: 0,
    first_token_millis: 0,
    inter_token_millis: 0,
    request_millis: 0,
  }
  let ticket : LunaOnlineRequestTicket = { lane: -1, epoch: 41UL }
  service.tickets.push(ticket)
  service.ticket_sequences.push(41UL)
  // This is the same post-commit timing seam used after lower admission and
  // dispatch succeed; receipt parsing is intentionally outside the interval.
  service.begin_request_timing_at(ticket, 100UL, true)
  service.publish_observation(LunaOnlineFramedAdmissionObservation, 0, 0, 0, 0)
  let accepted = stream.take_observation()
  server.record_retry_and_ack(accepted)

  service.publish_observation(
    LunaOnlineFramedTokenObservation,
    0,
    0,
    0,
    0,
    latency=service.sample_request_timing_at(
      ticket,
      token=true,
      terminal=false,
      observed=125UL,
      available=true,
    ),
  )
  let first = stream.take_observation()
  server.record_retry_and_ack(first)

  service.publish_observation(
    LunaOnlineFramedTokenObservation,
    0,
    0,
    0,
    0,
    latency=service.sample_request_timing_at(
      ticket,
      token=true,
      terminal=false,
      observed=145UL,
      available=true,
    ),
  )
  let second = stream.take_observation()
  server.record_retry_and_ack(second)

  service.publish_observation(
    LunaOnlineFramedCompletionObservation,
    0,
    0,
    0,
    0,
    latency=service.sample_request_timing_at(
      ticket,
      token=false,
      terminal=true,
      observed=180UL,
      available=true,
    ),
  )
  let terminal = stream.take_observation()
  server.record_retry_and_ack(terminal)

  assert_eq(server.first_token_samples, 1)
  assert_eq(server.inter_token_samples, 1)
  assert_eq(server.request_samples, 1)
  assert_eq(server.first_token_millis, 25)
  assert_eq(server.inter_token_millis, 20)
  assert_eq(server.request_millis, 80)

  service.tickets.clear()
  service.ticket_sequences.clear()
  let reused : LunaOnlineRequestTicket = { lane: -1, epoch: 42UL }
  service.tickets.push(reused)
  service.ticket_sequences.push(42UL)
  service.begin_request_timing_at(reused, 200UL, true)
  try terminal.latency_kind() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected stale observation after lane reuse")
  } noraise {
    _ => fail("lane reuse revived an ACKed latency observation")
  }
  assert_eq(server.first_token_samples, 1)
  assert_eq(server.inter_token_samples, 1)
  assert_eq(server.request_samples, 1)
  service.retire_request_timing(reused)
  service.tickets.clear()
  service.ticket_sequences.clear()
  retire_fixture_stream(stream)
}

///|
test "Luna usage observation backpressures and exposes only bounded counts" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let stream = observation_fixture_stream(compatibility)
  let owner = @luna_event.LunaEventOwner::new(coordinator_fixture_inference())
  owner.publish_usage(input_tokens=7, cached_input_tokens=2, output_tokens=3)
  let view = owner.view()
  assert_true(service.publish_event_observation(view))
  assert_eq(stream.progress(), LunaOnlineFramedCoordinatorObservationReady)
  assert_eq(stream.progress(), LunaOnlineFramedCoordinatorObservationReady)

  let observation = stream.take_observation()
  assert_eq(observation.kind(), LunaOnlineFramedUsageObservation)
  assert_eq(observation.input_tokens(), 7)
  assert_eq(observation.cached_input_tokens(), 2)
  assert_eq(observation.output_tokens(), 3)
  assert_eq(observation.total_tokens(), 10)
  observation.ack()
  try observation.ack() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected consumed observation error")
  } noraise {
    _ => fail("usage observation ACKed twice")
  }
  owner.retire(view)
  retire_fixture_stream(stream)
}

///|
test "Luna max-one coordinator explicitly consumes observation bookkeeping" {
  let coordinator = coordinator_fixture()
  coordinator.service.publish_observation(
    LunaOnlineFramedAdmissionObservation,
    0,
    0,
    0,
    0,
  )
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorAdvanced)
  assert_eq(coordinator.service.observation_kind, 0)
  assert_false(coordinator.service.observation_taken)
}

///|
test "Luna terminal observations preserve cancellation deadline and failure" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let stream = observation_fixture_stream(compatibility)
  let owner = @luna_event.LunaEventOwner::new(coordinator_fixture_inference())
  let tail = FixedArray::make(0, b'\x00')

  owner.publish_completed(
    Cancelled,
    input_tokens=3,
    cached_input_tokens=0,
    output_tokens=1,
    tail,
    tail_offset=0,
    tail_length=0,
  )
  let cancelled = owner.view()
  assert_true(service.publish_event_observation(cancelled))
  let cancellation = stream.take_observation()
  assert_eq(cancellation.kind(), LunaOnlineFramedCancellationObservation)
  try cancellation.input_tokens() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle) => ()
    _ => fail("unexpected non-usage count error")
  } noraise {
    _ => fail("terminal observation exposed usage counts")
  }
  cancellation.ack()
  owner.retire(cancelled)

  let code = FixedArray::make(1, b'x')
  owner.publish_failed(code, code_offset=0, code_length=1, retryable=false)
  let failed = owner.view()
  service.instance.failure_kind = DeadlineLatchedFailure
  assert_true(service.publish_event_observation(failed))
  let deadline = stream.take_observation()
  assert_eq(deadline.kind(), LunaOnlineFramedDeadlineObservation)
  deadline.ack()
  owner.retire(failed)

  owner.publish_failed(code, code_offset=0, code_length=1, retryable=false)
  let output_failed = owner.view()
  service.instance.failure_kind = OutputLatchedFailure
  assert_true(service.publish_event_observation(output_failed))
  let failure = stream.take_observation()
  assert_eq(failure.kind(), LunaOnlineFramedRequestFailureObservation)
  failure.ack()
  owner.retire(output_failed)
  retire_fixture_stream(stream)
}

///|
test "Luna worker failure precedes per-request terminal and exact restart" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let stream = observation_fixture_stream(compatibility)
  let owner = @luna_event.LunaEventOwner::new(coordinator_fixture_inference())
  let code = FixedArray::make(1, b'w')
  let ticket : LunaOnlineRequestTicket = { lane: -1, epoch: 51UL }
  service.tickets.push(ticket)
  service.ticket_sequences.push(51UL)
  service.begin_request_timing(ticket)
  service.publish_observation(LunaOnlineFramedAdmissionObservation, 0, 0, 0, 0)
  let admitted = stream.take_observation()
  assert_eq(admitted.kind(), LunaOnlineFramedAdmissionObservation)
  admitted.ack()
  service.publish_observation(
    LunaOnlineFramedWorkerFailureObservation,
    0,
    0,
    0,
    0,
  )
  let worker = stream.take_observation()
  assert_eq(worker.kind(), LunaOnlineFramedWorkerFailureObservation)
  worker.ack()

  service.instance.restart_observation_pending = true
  assert_true(service.instance.take_restart_observation())
  service.publish_observation(
    LunaOnlineFramedWorkerRestartObservation,
    0,
    0,
    0,
    0,
  )
  let restarted = stream.take_observation()
  assert_eq(restarted.kind(), LunaOnlineFramedWorkerRestartObservation)
  restarted.ack()

  owner.publish_usage(input_tokens=3, cached_input_tokens=0, output_tokens=1)
  let usage_event = owner.view()
  assert_true(service.publish_event_observation(usage_event))
  let usage = stream.take_observation()
  assert_eq(usage.kind(), LunaOnlineFramedUsageObservation)
  usage.ack()
  owner.retire(usage_event)

  owner.publish_failed(code, code_offset=0, code_length=1, retryable=true)
  let failed = owner.view()
  service.instance.failure_kind = WorkerLatchedFailure
  assert_true(service.publish_event_observation(failed))
  let failure = stream.take_observation()
  assert_eq(failure.kind(), LunaOnlineFramedWorkerRequestTerminalObservation)
  assert_eq(failure.latency_kind(), LunaOnlineFramedRequestLatency)
  assert_true(failure.latency_millis() >= 0)
  failure.ack()
  owner.retire(failed)
  service.tickets.clear()
  service.ticket_sequences.clear()
  retire_fixture_stream(stream)
}

///|
test "Luna timing pulse exposes only a relative authenticated scalar" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let stream = observation_fixture_stream(compatibility)
  service.publish_observation(LunaOnlineFramedTokenObservation, 0, 0, 0, 0, latency={
    kind: 2,
    millis: 17,
  })
  let observation = stream.take_observation()
  assert_eq(observation.kind(), LunaOnlineFramedTokenObservation)
  assert_eq(observation.latency_kind(), LunaOnlineFramedFirstTokenLatency)
  assert_eq(observation.latency_millis(), 17)
  observation.ack()
  try observation.latency_millis() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected stale timing observation error")
  } noraise {
    _ => fail("ACKed timing observation remained live")
  }
  retire_fixture_stream(stream)
}

///|
test "Luna disconnect drains observation before reused stream generation" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let first = observation_fixture_stream(compatibility)
  service.publish_observation(LunaOnlineFramedAdmissionObservation, 0, 0, 0, 0)
  let stale = first.take_observation()
  first.disconnect()
  assert_eq(first.progress(), LunaOnlineFramedCoordinatorObservationReady)
  assert_eq(stale.kind(), LunaOnlineFramedAdmissionObservation)
  stale.ack()
  retire_fixture_stream(first)
  try stale.kind() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected retired observation error")
  } noraise {
    _ => fail("retired observation remained live")
  }

  let opened = service.open_stream()
  let second = service.take_open_stream(opened)
  service.publish_observation(LunaOnlineFramedCompletionObservation, 0, 0, 0, 0)
  let current = second.take_observation()
  try stale.ack() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected cross-stream observation error")
  } noraise {
    _ => fail("old observation consumed successor pulse")
  }
  assert_eq(current.kind(), LunaOnlineFramedCompletionObservation)
  current.ack()
  retire_fixture_stream(second)
}
