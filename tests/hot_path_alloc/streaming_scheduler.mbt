///|
fn streaming_request() -> HotPathRequestFixture {
  let limits = inference_limits()
  let storage = try! @inference.LunaRequestSemanticStorage::new(
    limits,
    try! @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = try! storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=ReadWrite,
  )
  try! write.push_cache_scope_byte(b'x')
  let work = try! write.finish()
  while (try! work.progress()).is_pending() {

  }
  let semantics = try! work.take_lease()
  let request = try! @scheduler.TokenizedRequest::new_with_prefix(
    request_id=try! @inference.RequestId::new(2UL),
    model_identity=model_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1, 2, 3, 4, 5], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(1UL),
    stop_tokens=try! semantics.stop_token_view(),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=1,
      context_ceiling=32,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(1000UL, limits),
    ),
    tokenizer_digest=try! @tokenizer.TokenizerDigest::parse(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    prefix_semantic=try! semantics.prefix_view(),
  )
  { request, semantics, }
}

///|
fn streaming_command_cycle(
  owner : @scheduler.Scheduler,
  generation : Int,
  fixture : HotPathRequestFixture,
  bytes : FixedArray[Byte],
  cap : @streaming_ir.Capability,
) -> Unit {
  alloc_probe_begin()
  let spill = (try! owner.next_streaming_command(0UL)).unwrap()
  let control = @streaming_ir.Control::new(1UL)
  let decision = control.decide({ ..spill, sequence: 1UL, }, cap)
  if !(decision.action is BeginSpill) {
    alloc_probe_abort()
  }
  try! @streaming_ir.encode(
    { command: spill, status: Pending, host: spill.host, },
    bytes,
  )
  let decoded = try! @streaming_ir.decode(bytes, 64)
  if !decoded.command.same_command(spill) {
    alloc_probe_abort()
  }
  try! owner.accept_streaming_reply({
    command: spill,
    status: HostReady,
    host: { slot: 0, generation, },
  })
  if alloc_probe_check(alloc_probe_end(), 1) != 1 {
    alloc_probe_abort()
  }
  // Admission owns request validation; measure the residency effect path itself.
  let handle = try! owner.admit(fixture.request, 0UL)
  alloc_probe_begin()
  let restore = (try! owner.next_streaming_command(0UL)).unwrap()
  let prepared_allocations = alloc_probe_end()
  if prepared_allocations != 0UL {
    println("restore planning allocations: \{prepared_allocations}")
    alloc_probe_abort()
  }
  alloc_probe_begin()
  if !(restore.opcode is Restore) {
    alloc_probe_abort()
  }
  try! owner.accept_streaming_reply({
    command: restore,
    status: DeviceReady,
    host: restore.host,
  })
  let evict = (try! owner.next_streaming_command(0UL)).unwrap()
  try! owner.accept_streaming_reply({
    command: evict,
    status: Evicted,
    host: evict.host,
  })
  if alloc_probe_check(alloc_probe_end(), 1) != 1 {
    alloc_probe_abort()
  }
  ignore(try! owner.cancel(handle))
  ignore(try! owner.take_terminal_notice())
}

///|
fn verify_streaming_scheduler_allocations() -> Unit {
  let owner = scheduler(prefix=true)
  let cap = try! @streaming_ir.Capability::new(
    slots=2,
    host_bytes=128L,
    page_bytes=64L,
    page_count=10,
    reserve_pages=10,
    restore_ns=100L,
    recompute_ns=1000L,
    separate_host_memory=true,
  )
  try! owner.configure_streaming(cap, 1UL)
  let fixture = streaming_request()
  ignore(try! owner.admit(fixture.request, 0UL))
  for i in 0..<5 {
    let submitted = try! owner.submitted_plan(try! owner.build_next())
    let row = try! submitted.prefill_row(0)
    let writer = try! owner.completion_writer(submitted)
    if i == 4 {
      try! writer.append_final_prefill_completed(
        slot=try! row.completion_slot(),
        request_id=try! row.request_id(),
        request_generation=try! row.request_generation(),
        processed_tokens=1,
        token_id=7,
      )
    } else {
      try! writer.append_prefill_completed(
        slot=try! row.completion_slot(),
        request_id=try! row.request_id(),
        request_generation=try! row.request_generation(),
        processed_tokens=1,
      )
    }
    commit_or_abort(owner, submitted, try! writer.submit())
  }
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  let bytes = FixedArray::make(64, b'\x00')
  for generation in 1..<=10000 {
    streaming_command_cycle(owner, generation, fixture, bytes, cap)
  }
  if owner.free_page_count() != 9 {
    alloc_probe_abort()
  }
  try! fixture.semantics.release()
  println(
    "Streaming scheduler and IR: 10000 spill/restore/evict cycles without heap allocation",
  )
}
