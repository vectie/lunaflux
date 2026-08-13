///|
fn execution_test_frame() -> (
  @worker_wire.ValidatedPlanFrame,
  @worker_wire.CompletionFrameBuffer,
) raise {
  let startup = worker_test_startup()
  let limits = startup.worker_limits()
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=1,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let schedule = @worker_protocol.SchedulePlanBuffer::new(limits)
  schedule.append_final_prefill(
    request_id=@inference.RequestId::new(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[0],
    pages=[allocator.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(1UL),
    sample_index=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  let submitted = schedule.submit_wire(
    sequence_value=1UL,
    model_plan_generation_value=startup.model_generation_value(),
    token_budget=1,
    has_previous=false,
    previous_sequence_value=0UL,
    loaded_generation_value=startup.model_generation_value(),
  )
  let frame_owner = @worker_wire.PlanFrameBuffer::new(
    limits,
    startup.inference_limits(),
  )
  (
    frame_owner.encode_plan(submitted),
    @worker_wire.CompletionFrameBuffer::new(limits, startup.inference_limits()),
  )
}

///|
fn fake_execution_transaction(
  fail_at : Int,
  calls : Array[String],
) -> String raise DeviceWorkerError {
  let names = ["authenticate", "stage", "execute", "sample", "finish", "submit"]
  for index, name in names {
    calls.push(name)
    if fail_at == index {
      raise Invalid(ResourceState)
    }
  }
  "completion"
}

///|
fn fake_owner_execution(
  lifecycle : Ref[DeviceWorkerLifecycle],
  fail_at : Int,
  abort_fails : Bool,
  calls : Array[String],
) -> String raise DeviceWorkerError {
  if lifecycle.val != WorkerReady {
    raise InvalidLifecycle(lifecycle.val)
  }
  fake_execution_transaction(fail_at, calls) catch {
    primary => {
      lifecycle.val = WorkerFaulted
      calls.push("abort")
      if abort_fails {
        raise CompletionAbortFailed(
          primary~,
          abort=CompletionFrameWriterNotOpen,
        )
      }
      raise primary
    }
  }
}

///|
test "frame forwarding publishes only after finish" {
  let calls : Array[String] = []
  assert_eq(fake_execution_transaction(-1, calls), "completion")
  assert_eq(calls, [
    "authenticate", "stage", "execute", "sample", "finish", "submit",
  ])
}

///|
test "every entered failure aborts, faults, and permits no retry" {
  let names = ["authenticate", "stage", "execute", "sample", "finish", "submit"]
  for fail_at in 0..<names.length() {
    let lifecycle = Ref(WorkerReady)
    let calls : Array[String] = []
    try fake_owner_execution(lifecycle, fail_at, false, calls) catch {
      Invalid(ResourceState) => ()
      _ => fail("unexpected forwarding failure")
    } noraise {
      _ => fail("failed execution returned a completion")
    }
    let expected : Array[String] = []
    for index in 0..<=fail_at {
      expected.push(names[index])
    }
    expected.push("abort")
    assert_eq(calls, expected)
    assert_eq(lifecycle.val, WorkerFaulted)
    try fake_owner_execution(lifecycle, -1, false, calls) catch {
      InvalidLifecycle(WorkerFaulted) => ()
      _ => fail("unexpected faulted retry error")
    } noraise {
      _ => fail("faulted worker executed again")
    }
    assert_eq(calls, expected)
  }
}

///|
test "finish failure cannot publish and real writer remains reusable" {
  let (frame, completions) = execution_test_frame()
  let writer = completions.writer_for(frame)
  writer.validate_plan(frame)
  writer.append_final_prefill_completed(frame.prefill_row(0), 0)
  try completions.current() catch {
    StaleFrame => ()
    _ => fail("unexpected in-progress completion state")
  } noraise {
    _ => fail("sampled completion published before finish")
  }
  let owner : DeviceWorkerOwner = {
    startup: worker_test_startup(),
    resources: empty_worker_resources(),
    lifecycle: WorkerReady,
  }
  try {
    let _ : Unit = fail_execution(owner, writer, Invalid(ResourceState))
  } catch {
    Invalid(ResourceState) => ()
    _ => fail("unexpected finish failure")
  } noraise {
    _ => fail("finish failure returned a completion")
  }
  assert_eq(owner.lifecycle(), WorkerFaulted)
  try completions.current() catch {
    StaleFrame => ()
    _ => fail("unexpected aborted completion state")
  } noraise {
    _ => fail("aborted completion remained observable")
  }
  let retry = completions.writer_for(frame)
  retry.append_final_prefill_completed(frame.prefill_row(0), 0)
  ignore(retry.submit())
}

///|
test "entered resource failure aborts a real writer and remains closeable" {
  let (frame, completions) = execution_test_frame()
  let writer = completions.writer_for(frame)
  let owner : DeviceWorkerOwner = {
    startup: worker_test_startup(),
    resources: empty_worker_resources(),
    lifecycle: WorkerReady,
  }
  try owner.execute_frame(frame, writer) catch {
    Invalid(ResourceState) => ()
    _ => fail("unexpected entered execution error")
  } noraise {
    _ => fail("incomplete worker executed")
  }
  assert_eq(owner.lifecycle(), WorkerFaulted)
  let retry = completions.writer_for(frame)
  retry.abort()
  owner.close()
  assert_eq(owner.lifecycle(), WorkerClosed)
}

///|
test "wrong lifecycle rejects before touching writer" {
  let (frame, completions) = execution_test_frame()
  let writer = completions.writer_for(frame)
  let owner : DeviceWorkerOwner = {
    startup: worker_test_startup(),
    resources: empty_worker_resources(),
    lifecycle: WorkerFaulted,
  }
  try owner.execute_frame(frame, writer) catch {
    InvalidLifecycle(WorkerFaulted) => ()
    _ => fail("unexpected wrong-lifecycle error")
  } noraise {
    _ => fail("faulted worker executed")
  }
  writer.append_final_prefill_completed(frame.prefill_row(0), 0)
  ignore(writer.submit())
  assert_eq(owner.lifecycle(), WorkerFaulted)
}

///|
test "abort double failure retains primary and cleanup errors" {
  let (frame, completions) = execution_test_frame()
  let writer = completions.writer_for(frame)
  writer.abort()
  let owner : DeviceWorkerOwner = {
    startup: worker_test_startup(),
    resources: empty_worker_resources(),
    lifecycle: WorkerReady,
  }
  try {
    let _ : Unit = fail_execution(owner, writer, Invalid(ResourceState))
  } catch {
    CompletionAbortFailed(
      primary=Invalid(ResourceState),
      abort=CompletionFrameWriterNotOpen
    ) => ()
    _ => fail("unexpected combined abort error")
  } noraise {
    _ => fail("double failure returned a completion")
  }
  assert_eq(owner.lifecycle(), WorkerFaulted)
}

///|
test "faulted owner is unready but remains closeable" {
  let owner : DeviceWorkerOwner = {
    startup: worker_test_startup(),
    resources: empty_worker_resources(),
    lifecycle: WorkerFaulted,
  }
  try owner.readiness_contract() catch {
    InvalidLifecycle(WorkerFaulted) => ()
    _ => fail("unexpected faulted readiness error")
  } noraise {
    _ => fail("faulted worker remained ready")
  }
  owner.close()
  assert_eq(owner.lifecycle(), WorkerClosed)
  owner.close()
}
