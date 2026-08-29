///|
fn phase23_balance_semantic_lease() -> @inference.LunaRequestSemanticLease raise {
  let limits = inference_limits()
  let storage = @inference.LunaRequestSemanticStorage::new(
    limits,
    @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let scope = b"balance"
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=scope.length(),
    cache_permission=Disabled,
  )
  for byte in scope {
    write.push_cache_scope_byte(byte)
  }
  let work = write.finish()
  while work.progress().is_pending() {

  }
  work.take_lease()
}

///|
fn phase23_balance_semantic_leases() -> FixedArray[
  @inference.LunaRequestSemanticLease?,
] raise {
  let leases = FixedArray::make(PHASE23_BALANCE_WAVE_SIZE, None)
  for index in 0..<leases.length() {
    leases[index] = Some(phase23_balance_semantic_lease())
  }
  leases
}

///|
fn phase23_balance_stop_view(
  leases : FixedArray[@inference.LunaRequestSemanticLease?],
  index : Int,
) -> @inference.LunaRequestStopTokenView raise {
  match leases[index] {
    Some(lease) => lease.stop_token_view()
    None => raise Failure("balance semantic lease slot was empty")
  }
}

///|
fn phase23_balance_request(
  id : UInt64,
  stop_tokens : @inference.LunaRequestStopTokenView,
  deadline_budget : UInt64,
) -> @scheduler.TokenizedRequest raise {
  let limits = inference_limits()
  @scheduler.TokenizedRequest::new(
    request_id=@inference.RequestId::new(id),
    model_identity=identity(),
    input_tokens=@inference.TokenBuffer::new([1], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(id),
    stop_tokens~,
    stream_preference=Streaming,
    effective_limits=@inference.EffectiveLimits::new(
      max_new_tokens=1,
      context_ceiling=4,
      limits,
    ),
    admission_deadline=@inference.AdmissionDeadline::from_receipt(
      0UL,
      @inference.DeadlineBudget::from_milliseconds(deadline_budget, limits),
    ),
  )
}

///|
fn phase23_balance_index(id : @inference.RequestId) -> Int raise {
  let value = id.value()
  if value == 0UL || value > PHASE23_BALANCE_REQUEST_COUNT.to_uint64() {
    raise Failure("balance publication request ID escaped the admitted range")
  }
  (value - 1UL).to_int()
}

///|
fn phase23_expected_reason(index : Int) -> @scheduler.TerminalReason {
  match index % PHASE23_BALANCE_WAVE_SIZE {
    6 | 7 => CancelledByCaller
    8 | 9 => DeadlineReached
    _ => MaximumOutputReached
  }
}

///|
fn phase23_drain_publications(
  service : @worker_service.WorkerService,
  ledger : FixedArray[Int],
) -> Int raise {
  let mut terminals = 0
  while service.generated_token_count() + service.terminal_notice_count() > 0 {
    let publication = service.take_next_publication()
    match publication.kind() {
      GeneratedTokenPublication => {
        let notice = publication.generated_token_notice()
        let index = phase23_balance_index(notice.request_id())
        if index % PHASE23_BALANCE_WAVE_SIZE >= 6 || (ledger[index] & 1) != 0 {
          raise Failure("balance generated token was unexpected or duplicated")
        }
        assert_eq(notice.position(), 0)
        ledger[index] = ledger[index] | 1
      }
      TerminalPublication => {
        let notice = publication.terminal_notice()
        let index = phase23_balance_index(notice.request_id())
        if (ledger[index] & 2) != 0 {
          raise Failure("balance terminal publication was duplicated")
        }
        assert_eq(notice.reason(), phase23_expected_reason(index))
        ledger[index] = ledger[index] | 2
        terminals += 1
      }
    }
  }
  terminals
}

///|
fn phase23_assert_balanced(
  service : @worker_service.WorkerService,
) -> Unit raise {
  assert_eq(service.outstanding_count(), 0)
  assert_eq(service.live_request_count(), 0)
  assert_eq(service.generated_token_count(), 0)
  assert_eq(service.terminal_notice_count(), 0)
  assert_eq(service.allocated_block_table_count(), 0)
  assert_eq(service.free_page_count(), PHASE23_BALANCE_PAGE_COUNT)
}

///|
fn phase23_run_wave(
  service : @worker_service.WorkerService,
  leases : FixedArray[@inference.LunaRequestSemanticLease?],
  ledger : FixedArray[Int],
  first_index : Int,
) -> Unit raise {
  for offset in 0..<2 {
    let id = (first_index + offset + 1).to_uint64()
    ignore(
      service.admit(
        phase23_balance_request(
          id,
          phase23_balance_stop_view(leases, offset),
          1000UL,
        ),
        0UL,
      ),
    )
  }
  guard service.progress() is ServiceStarted(_) else {
    raise Failure("balance wave did not start its first two-row plan")
  }
  for offset in 2..<4 {
    let id = (first_index + offset + 1).to_uint64()
    ignore(
      service.admit(
        phase23_balance_request(
          id,
          phase23_balance_stop_view(leases, offset),
          1000UL,
        ),
        0UL,
      ),
    )
  }
  guard service.progress() is ServicePlanPending(_) else {
    raise Failure("balance wave did not pin its second plan")
  }
  for offset in 4..<6 {
    let id = (first_index + offset + 1).to_uint64()
    ignore(
      service.admit(
        phase23_balance_request(
          id,
          phase23_balance_stop_view(leases, offset),
          1000UL,
        ),
        0UL,
      ),
    )
  }
  let cancel_first = service.admit(
    phase23_balance_request(
      (first_index + 7).to_uint64(),
      phase23_balance_stop_view(leases, 6),
      1000UL,
    ),
    0UL,
  )
  let cancel_second = service.admit(
    phase23_balance_request(
      (first_index + 8).to_uint64(),
      phase23_balance_stop_view(leases, 7),
      1000UL,
    ),
    0UL,
  )
  for offset in 8..<10 {
    let id = (first_index + offset + 1).to_uint64()
    ignore(
      service.admit(
        phase23_balance_request(
          id,
          phase23_balance_stop_view(leases, offset),
          1UL,
        ),
        0UL,
      ),
    )
  }
  service.cancel(cancel_first)
  service.cancel(cancel_second)
  assert_eq(service.expire_deadlines(1UL), 2)
  assert_eq(service.terminal_notice_count(), 4)

  let mut terminals = 0
  let mut saw_completion_ready = false
  let mut saw_completion_backpressure = false
  let mut attempts = 0
  while terminals < PHASE23_BALANCE_WAVE_SIZE && attempts < 1000000 {
    match service.progress() {
      ServiceCompletionReady(_) => saw_completion_ready = true
      ServiceBackpressured => {
        if !saw_completion_ready {
          raise Failure("balance wave backpressured before a completion")
        }
        saw_completion_backpressure = true
      }
      ServiceIdle if terminals < PHASE23_BALANCE_WAVE_SIZE =>
        raise Failure("balance wave idled before exact reconciliation")
      ServiceIdle
      | ServiceAdvanced
      | ServicePlanPending(_)
      | ServiceStarted(_)
      | ServicePending(_)
      | ServiceCommitted(_) => ()
    }
    if saw_completion_backpressure {
      terminals += phase23_drain_publications(service, ledger)
    }
    attempts += 1
  }
  assert_true(saw_completion_backpressure)
  assert_eq(terminals, PHASE23_BALANCE_WAVE_SIZE)
  for index in first_index..<(first_index + PHASE23_BALANCE_WAVE_SIZE) {
    let expected = if index % PHASE23_BALANCE_WAVE_SIZE < 6 { 3 } else { 2 }
    assert_eq(ledger[index], expected)
  }
  phase23_assert_balanced(service)
}

///|
fn phase23_recover_balanced_worker(
  service : @worker_service.WorkerService,
) -> Unit raise {
  phase23_assert_balanced(service)
  service.begin_recovery()
  assert_eq(service.lifecycle(), ServiceDeviceInvalidationRequired)
  assert_eq(service.invalidate_device_state(), 0)
  match service.restart() {
    Restarted => ()
    RestartCleanupRequired =>
      raise Failure("balance replacement worker required cleanup")
  }
  phase23_assert_balanced(service)
}

///|
async fn phase23_shutdown_balanced_worker(
  service : @worker_service.WorkerService,
) -> Unit {
  service.begin_shutdown_maintenance()
  let mut attempts = 0
  while attempts < 1000000 {
    match service.progress_maintenance() {
      LunaWorkerServiceMaintenancePending => {
        attempts += 1
        @async.sleep(1)
      }
      LunaWorkerServiceMaintenanceAdvanced => attempts += 1
      LunaWorkerServiceMaintenanceCleanupStuck => {
        attempts += 1
        @async.sleep(1)
      }
      LunaWorkerServiceClosed => {
        assert_eq(service.lifecycle(), ServiceClosed)
        return
      }
      LunaWorkerServiceMaintenanceCleanupRequired
      | LunaWorkerServiceRecoveryReady
      | LunaWorkerServiceFailCloseReady
      | LunaWorkerServiceRestartReady =>
        raise Failure("balance shutdown entered a non-clean disposition")
    }
  }
  raise Failure("balance shutdown did not cooperatively close and reap")
}

///|
async fn run_phase23_balance_10000(
  executable : Bytes,
  fixture_root : StringView,
) -> Unit {
  let service = phase23_balance_service(executable, fixture_root)
  let semantic_leases = phase23_balance_semantic_leases()
  let ledger = FixedArray::make(PHASE23_BALANCE_REQUEST_COUNT, 0)
  let wave_count = PHASE23_BALANCE_REQUEST_COUNT / PHASE23_BALANCE_WAVE_SIZE
  for wave in 0..<wave_count {
    let mut wave_complete = false
    defer (if !wave_complete {
      println("LunaFlux Phase 2/3 balance wave=\{wave} failure")
    })
    phase23_run_wave(
      service,
      semantic_leases,
      ledger,
      wave * PHASE23_BALANCE_WAVE_SIZE,
    )
    wave_complete = true
    if (wave + 1) % 100 == 0 {
      phase23_recover_balanced_worker(service)
    }
  }
  for index in 0..<PHASE23_BALANCE_REQUEST_COUNT {
    let expected = if index % PHASE23_BALANCE_WAVE_SIZE < 6 { 3 } else { 2 }
    assert_eq(ledger[index], expected)
  }
  phase23_assert_balanced(service)
  phase23_shutdown_balanced_worker(service)
  for index in 0..<semantic_leases.length() {
    match semantic_leases[index] {
      Some(lease) => lease.release()
      None => raise Failure("balance semantic lease disappeared")
    }
    semantic_leases[index] = None
  }
  println("LunaFlux Phase 2/3 10,000-request balance evidence passed.")
}
