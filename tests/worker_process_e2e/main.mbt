///|
fn worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=4,
    max_decode_rows=4,
    max_plan_rows=8,
    max_plan_tokens=32,
    max_plan_pages=32,
    max_capabilities=32,
    max_completion_slots=8,
    max_sequence_tokens=64,
    max_token_id=1024,
  )
}

///|
fn inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=1024,
    max_input_tokens=64,
    max_new_tokens=64,
    max_context_tokens=64,
    max_token_id=1024,
    max_stop_token_ids=8,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=64,
    max_cache_scope_bytes=64,
    max_decoded_delta_bytes=64,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn page_allocator() -> @page_allocator.PageAllocator raise {
  @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=8,
      max_run_pages=8,
      max_active_references_per_page=8,
      max_cached_references_per_page=8,
    ),
  )
}

///|
fn plan(
  sequence : UInt64,
  request : UInt64,
) -> @worker_protocol.SubmittedSchedulePlan raise {
  let limits = worker_limits()
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  owner.append_prefill(
    request_id=@inference.RequestId::new(request),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1, 2],
    pages=[page_allocator().allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  owner.submit_wire(
    sequence_value=sequence,
    model_plan_generation_value=1UL,
    token_budget=2,
    has_previous=sequence > 1UL,
    previous_sequence_value=if sequence > 1UL { sequence - 1UL } else { 0UL },
    loaded_generation_value=1UL,
  )
}

///|
fn run(executable : Bytes) -> Unit raise {
  let supervisor = @worker_process.start(
    executable,
    worker_limits(),
    inference_limits(),
    @worker_process.WorkerProcessLimits::new(
      max_frame_bytes=4096,
      io_timeout_millis=5000,
      shutdown_timeout_millis=5000,
    ),
    has_predecessor=false,
    predecessor_value=0UL,
  )
  let plan_a1 = plan(1UL, 11UL)
  let plan_b2 = plan(2UL, 12UL)
  let submission_a1 = supervisor.submit(plan_a1)
  let submission_b2 = supervisor.submit(plan_b2)
  let completion_a1 = supervisor.receive(submission_a1, plan_a1)
  let frame_a1 = completion_a1.frame(supervisor)
  assert_eq(frame_a1.entry_count(), 1)
  assert_eq(frame_a1.entry_for_slot(0).processed_tokens(), 2)
  supervisor.retire_received(completion_a1)
  let plan_a3 = plan(3UL, 13UL)
  let submission_a3 = supervisor.submit(plan_a3)
  let completion_b2 = supervisor.receive(submission_b2, plan_b2)
  supervisor.retire_received(completion_b2)
  let completion_a3 = supervisor.receive(submission_a3, plan_a3)
  supervisor.retire_received(completion_a3)
  assert_eq(supervisor.outstanding_count(), 0)
  supervisor.shutdown_clean()
  supervisor.close()
  assert_true(supervisor.lifecycle() == Closed)
}

///|
fn main {
  match @env.args() {
    [_, executable] => try! run(@utf8.encode(executable))
    _ => println("worker executable path is required")
  }
}
