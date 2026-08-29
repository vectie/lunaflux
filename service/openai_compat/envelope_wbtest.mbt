///|
fn openai_max_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=16384,
    max_input_tokens=16384,
    max_new_tokens=16384,
    max_context_tokens=32768,
    max_token_id=1048575,
    max_stop_token_ids=64,
    max_stop_strings=64,
    max_stop_string_bytes=256,
    max_trace_bytes=256,
    max_cache_scope_bytes=256,
    max_decoded_delta_bytes=16384,
    max_deadline_millis=86400000UL,
    max_top_k=1048576,
    max_temperature=100.0,
  )
}

///|
test "Luna OpenAI checked sizing covers the exact maximum envelope" {
  let limits = openai_max_limits()
  let expected = 2048UL +
    (16384UL + 64UL + 256UL) * 6UL +
    64UL +
    20UL +
    256UL +
    64UL
  assert_eq(LunaOpenAICompatWorkspace::required_byte_cells(limits), expected)
  assert_eq(LunaOpenAICompatWorkspace::required_reference_cells(limits), 1UL)
  let model : FixedArray[Byte] = FixedArray::make(256, b'm')
  let prefix : FixedArray[Byte] = FixedArray::make(64, b'r')
  let workspace = LunaOpenAICompatWorkspace::new(
    limits,
    LunaOpenAICompatStepBudget::new(65536),
    model,
    model_alias_offset=0,
    model_alias_length=256,
    prefix,
    response_id_prefix_offset=0,
    response_id_prefix_length=64,
  )
  let payload : FixedArray[Byte] = FixedArray::make(16384, 0)
  let owner = @luna_event.LunaEventOwner::new(limits)
  assert_eq(
    owner.publish_token_status(
      token_id=1048575,
      position=16383,
      payload,
      source_offset=0,
      length=payload.length(),
    ),
    LunaEventPublishReady,
  )
  let event = owner.view()
  let work = workspace.begin(
    event,
    LunaOpenAIChatCompletions,
    response_sequence=18446744073709551615UL,
  )
  openai_drive_ready(work)
  let view = work.take_view()
  assert_true(view.length().to_uint64() <= expected - 320UL)
  let destination : FixedArray[Byte] = FixedArray::make(view.length(), 0)
  let mut offset = 0
  while offset < view.length() {
    let remaining = view.length() - offset
    let length = if remaining < 65536 { remaining } else { 65536 }
    assert_eq(
      view.copy_chunk_to(
        destination,
        destination_offset=offset,
        source_offset=offset,
        length~,
      ),
      length,
    )
    offset += length
  }
  view.release()
  owner.retire(event)
}

///|
test "Luna OpenAI startup copies aliases and rejects malformed ranges" {
  let limits = openai_wb_fixture_limits()
  let model : FixedArray[Byte] = [b'm', b'o', b'd', b'e', b'l']
  let prefix : FixedArray[Byte] = [b'i', b'd', b'_']
  let workspace = LunaOpenAICompatWorkspace::new(
    limits,
    LunaOpenAICompatStepBudget::new(1),
    model,
    model_alias_offset=0,
    model_alias_length=model.length(),
    prefix,
    response_id_prefix_offset=0,
    response_id_prefix_length=prefix.length(),
  )
  model[0] = b'x'
  prefix[0] = b'x'
  assert_eq(workspace.model_alias[0], b'm')
  assert_eq(workspace.response_id_prefix[0], b'i')
  openai_expect_rule(
    () => {
      ignore(
        LunaOpenAICompatWorkspace::new(
          limits,
          LunaOpenAICompatStepBudget::new(1),
          model,
          model_alias_offset=2147483647,
          model_alias_length=1,
          prefix,
          response_id_prefix_offset=0,
          response_id_prefix_length=1,
        ),
      )
    },
    LunaOpenAICompatModelAlias,
  )
  let malformed : FixedArray[Byte] = [0xf0, 0x80, 0x80, 0x80]
  openai_expect_rule(
    () => {
      ignore(
        LunaOpenAICompatWorkspace::new(
          limits,
          LunaOpenAICompatStepBudget::new(1),
          malformed,
          model_alias_offset=0,
          model_alias_length=4,
          prefix,
          response_id_prefix_offset=0,
          response_id_prefix_length=1,
        ),
      )
    },
    LunaOpenAICompatModelAlias,
  )
  let unsafe_prefix : FixedArray[Byte] = [b'"']
  openai_expect_rule(
    () => {
      ignore(
        LunaOpenAICompatWorkspace::new(
          limits,
          LunaOpenAICompatStepBudget::new(1),
          model,
          model_alias_offset=0,
          model_alias_length=1,
          unsafe_prefix,
          response_id_prefix_offset=0,
          response_id_prefix_length=1,
        ),
      )
    },
    LunaOpenAICompatResponseId,
  )
}

///|
test "Luna OpenAI limits and step budgets reject outside the frozen envelope" {
  for invalid in [0, 65537] {
    openai_expect_rule(
      () => ignore(LunaOpenAICompatStepBudget::new(invalid)),
      LunaOpenAICompatWorkUnits,
    )
  }
}

///|
test "Luna OpenAI response sequence rejects zero and accepts UInt64 maximum" {
  let limits = openai_wb_fixture_limits()
  let (owner, event) = openai_token_event(limits)
  let workspace = openai_wb_workspace(limits, 1)
  openai_expect_rule(
    () => {
      ignore(workspace.begin(event, LunaOpenAIResponses, response_sequence=0UL))
    },
    LunaOpenAICompatResponseId,
  )
  assert_eq(workspace.phase, LUNA_OPENAI_COMPAT_IDLE)
  let work = workspace.begin(
    event,
    LunaOpenAIResponses,
    response_sequence=18446744073709551615UL,
  )
  openai_drive_ready(work)
  work.abort()
  owner.retire(event)
}
