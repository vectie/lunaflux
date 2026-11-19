///|
async fn fixture() -> (@model.Spec, @text.Program) {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(spec, maximum_rows=4)
  let compilation = @attention.compile(
    operators,
    device_target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    target_id=@strategy.AttentionTargetId::new(
      "sm121-unqualified-software-profile",
    ),
    capabilities=@strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=49152L,
      max_pipeline_stages=1,
      parallel_workgroup_target=1,
      supports_async_copy=false,
      matrix_shape=Some(
        @strategy.AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
      ),
    ),
    maximum_context_tokens=65,
  )
  let stack = @decoder.Program::new(operators, compilation)
  (spec, @text.Program::new(operators, stack, maximum_workspace_bytes=1048576L))
}

///|
fn word(
  host : @minicpm_v47_request.Host,
  port : @minicpm_v47_request.Port,
  index~ : Int32 = 0,
) -> Int32 {
  let packet = host.packet(port)
  let start = index * 4
  (packet[start].to_uint() |
  (packet[start + 1].to_uint() << 8) |
  (packet[start + 2].to_uint() << 16) |
  (packet[start + 3].to_uint() << 24)).reinterpret_as_int()
}

///|
async test "chunked prefill selects only final row then decode feeds prior output" {
  let (spec, program) = fixture()
  let profile = @minicpm_v47_request.Profile::from_program(program)
  assert_eq(
    (profile.maximum_rows(), profile.context_tokens(), profile.page_capacity()),
    (4, 65, 5),
  )
  let host = @minicpm_v47_request.Host::new(profile)
  let prompt = @prompt.prepare_token_prompt(
    spec,
    Array::makei(17, i => 1000 + i),
    maximum_tokens=65,
  )
  host.begin(
    prompt,
    maximum_new_tokens=8,
    stop_token_ids=ReadOnlyArray::from_array(@prompt.stop_token_ids()),
  )
  for chunk in 0..<5 {
    assert_true(host.stage_next())
    let rows = if chunk == 4 { 1 } else { 4 }
    assert_eq(host.staged_phase(), @attention.Phase::Prefill)
    assert_eq(word(host, Counts), 1)
    assert_eq(word(host, Counts, index=1), 0)
    assert_eq(word(host, Counts, index=2), 1)
    assert_eq(word(host, Counts, index=3), rows)
    assert_eq(word(host, Counts, index=4), if chunk == 4 { 2 } else { 1 })
    assert_eq(word(host, Reset), if chunk == 0 { 1 } else { 0 })
    assert_eq(word(host, RowOffsets, index=1), rows)
    assert_eq(word(host, PageOffsets, index=1), if chunk == 4 { 2 } else { 1 })
    assert_eq(word(host, SequenceLength), chunk * 4 + rows)
    assert_eq(word(host, TokenIds), 1000 + chunk * 4)
    assert_eq(word(host, CausalPositions), chunk * 4)
    assert_eq(word(host, CanvasPositions), chunk * 4)
    assert_eq(word(host, SelectedRows), rows - 1)
    assert_eq(word(host, OutputCount), if chunk == 4 { 1 } else { 0 })
    assert_eq(host.live_bytes(TokenIds), rows * 4)
    assert_eq(host.live_bytes(CanvasPositions), rows * 12)
    assert_eq(host.cached_tokens(), chunk * 4)
    host.mark_submitted()
    // The ignored intermediate value proves no sample is consumed there.
    host.complete_step(if chunk == 4 { 120 } else { -1 })
  }
  assert_eq(host.cached_tokens(), 17)
  assert_eq(host.emitted_tokens(), [120])
  assert_true(host.stage_next())
  assert_eq(host.staged_phase(), @attention.Phase::Decode)
  assert_eq(word(host, Counts), 0)
  assert_eq(word(host, Counts, index=1), 1)
  assert_eq(word(host, TokenIds), 120)
  assert_eq(word(host, CausalPositions), 17)
  assert_eq(word(host, CanvasPositions, index=2), 17)
  assert_eq(word(host, SequenceLength), 18)
  assert_eq(word(host, OutputCount), 1)
  host.mark_submitted()
  host.complete_step(248046)
  assert_true(host.is_finished())
  assert_eq(host.finish_reason(), Some(@inference.FinishReason::StopSequence))
  assert_eq(host.sampled_tokens(), 2)
  assert_eq(host.emitted_tokens(), [120])
  assert_true(!host.stage_next())
}

///|
async test "context ceiling is exact and final prefill can sample at the ceiling" {
  let (spec, program) = fixture()
  let host = @minicpm_v47_request.Host::new(
    @minicpm_v47_request.Profile::from_program(program),
  )
  let prompt = @prompt.prepare_token_prompt(
    spec,
    [10, 11, 12],
    maximum_tokens=65,
  )
  host.begin(
    prompt,
    maximum_new_tokens=10,
    stop_token_ids=[],
    context_ceiling=4,
  )
  assert_true(host.stage_next())
  host.mark_submitted()
  host.complete_step(42)
  assert_true(!host.is_finished())
  assert_true(host.stage_next())
  host.mark_submitted()
  host.complete_step(43)
  assert_eq(host.cached_tokens(), 4)
  assert_eq(host.emitted_tokens(), [42, 43])
  assert_eq(host.finish_reason(), Some(@inference.FinishReason::TokenLimit))
  host.begin(
    prompt,
    maximum_new_tokens=10,
    stop_token_ids=[],
    context_ceiling=3,
  )
  assert_true(host.stage_next())
  assert_eq(word(host, Reset), 1)
  host.mark_submitted()
  host.complete_step(44)
  assert_eq(host.emitted_tokens(), [44])
  assert_eq(host.finish_reason(), Some(@inference.FinishReason::TokenLimit))
}

///|
async test "requested token limit stops before an extra decode input" {
  let (spec, program) = fixture()
  let host = @minicpm_v47_request.Host::new(
    @minicpm_v47_request.Profile::from_program(program),
  )
  let prompt = @prompt.prepare_token_prompt(spec, [10], maximum_tokens=65)
  host.begin(prompt, maximum_new_tokens=1, stop_token_ids=[])
  assert_true(host.stage_next())
  host.mark_submitted()
  host.complete_step(9)
  assert_eq(host.cached_tokens(), 1)
  assert_eq(host.sampled_tokens(), 1)
  assert_eq(host.emitted_tokens(), [9])
  assert_eq(host.finish_reason(), Some(@inference.FinishReason::TokenLimit))
  assert_true(!host.stage_next())
}

///|
async test "lifecycle invalid sample cancellation and new request reset" {
  let (spec, program) = fixture()
  let host = @minicpm_v47_request.Host::new(
    @minicpm_v47_request.Profile::from_program(program),
  )
  let prompt = @prompt.prepare_token_prompt(spec, [10, 11], maximum_tokens=65)
  assert_true(
    try host.stage_next() catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  host.begin(prompt, maximum_new_tokens=3, stop_token_ids=[])
  assert_true(host.stage_next())
  assert_true(
    try host.complete_step(10) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  host.mark_submitted()
  assert_true(
    try host.stage_next() catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try host.complete_step(-1) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_eq(host.cached_tokens(), 0)
  host.begin(prompt, maximum_new_tokens=3, stop_token_ids=[])
  assert_true(host.stage_next())
  assert_eq(word(host, Reset), 1)
  host.mark_submitted()
  host.complete_step(12)
  assert_true(
    try host.complete_step(13) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  host.stop_quiescent(Cancelled)
  assert_eq(host.finish_reason(), Some(@inference.FinishReason::Cancelled))
  host.begin(prompt, maximum_new_tokens=3, stop_token_ids=[])
  assert_true(host.stage_next())
  assert_eq(word(host, Reset), 1)
  assert_eq(word(host, PhysicalPages, index=4), 4)
  assert_eq(word(host, RequestSlots), 0)
  assert_eq(host.emitted_tokens(), [])
}

///|
async test "request admission rejects context overflow and invalid native token IDs" {
  let (spec, program) = fixture()
  let host = @minicpm_v47_request.Host::new(
    @minicpm_v47_request.Profile::from_program(program),
  )
  let prompt = @prompt.prepare_token_prompt(spec, [10, 11], maximum_tokens=65)
  assert_true(
    try host.begin(prompt, maximum_new_tokens=0, stop_token_ids=[]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      host.begin(
        prompt,
        maximum_new_tokens=2,
        stop_token_ids=[],
        context_ceiling=1,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  // Capacity is 65, although the final physical page has room through token 80.
  assert_true(
    try
      host.begin(
        prompt,
        maximum_new_tokens=2,
        stop_token_ids=[],
        context_ceiling=66,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      host.begin(prompt, maximum_new_tokens=2, stop_token_ids=[248144])
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try @prompt.prepare_token_prompt(spec, [-1], maximum_tokens=65) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try @prompt.prepare_token_prompt(spec, [248144], maximum_tokens=65) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try @prompt.prepare_token_prompt(spec, [], maximum_tokens=65) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let runner = @minicpm_v47_request.Runner::new(program)
  runner.close()
  runner.close()
  assert_true(
    try runner.begin(prompt, maximum_new_tokens=2, stop_token_ids=[]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
