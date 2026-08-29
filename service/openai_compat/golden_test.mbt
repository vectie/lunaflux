///|
fn openai_fixture_limits(
  max_delta? : Int = 32,
) -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=16,
    max_new_tokens=12,
    max_context_tokens=32,
    max_token_id=255,
    max_stop_token_ids=4,
    max_stop_strings=3,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=max_delta,
    max_deadline_millis=10000UL,
    max_top_k=256,
    max_temperature=8.0,
  )
}

///|
fn openai_fixture_identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn openai_workspace(
  limits : @inference.InferenceLimits,
  budget : Int,
  model? : FixedArray[Byte] = [b'm', b'o', b'd', b'e', b'l', b'-', b'x'],
) -> LunaOpenAICompatWorkspace raise {
  let prefix : FixedArray[Byte] = [b'r', b'e', b's', b'p', b'_']
  LunaOpenAICompatWorkspace::new(
    limits,
    LunaOpenAICompatStepBudget::new(budget),
    model,
    model_alias_offset=0,
    model_alias_length=model.length(),
    prefix,
    response_id_prefix_offset=0,
    response_id_prefix_length=prefix.length(),
  )
}

///|
fn openai_encode(
  workspace : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
  mode : LunaOpenAICompatMode,
  sequence : UInt64,
) -> (FixedArray[Byte], UInt64) raise {
  let work = workspace.begin(event, mode, response_sequence=sequence)
  while work.state() == LunaOpenAICompatPending {
    assert_eq(work.progress(), LunaOpenAICompatAdvanced)
    assert_true(work.last_work_units() > 0)
  }
  assert_eq(work.state(), LunaOpenAICompatReady)
  let total = work.total_work_units()
  let view = work.take_view()
  let output : FixedArray[Byte] = FixedArray::make(view.length(), 0)
  let mut offset = 0
  while offset < output.length() {
    let length = 1
    assert_eq(
      view.copy_chunk_to(
        output,
        destination_offset=offset,
        source_offset=offset,
        length~,
      ),
      length,
    )
    offset += length
  }
  view.release()
  (output, total)
}

///|
fn openai_assert_bytes(
  actual : FixedArray[Byte],
  expected : Bytes,
) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for index in 0..<actual.length() {
    assert_eq(actual[index], expected[index])
  }
}

///|
test "OpenAI Responses and Chat Accepted goldens are canonical SSE" {
  let limits = openai_fixture_limits()
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=4,
    context_ceiling=16,
    limits,
  )
  let modes : FixedArray[LunaOpenAICompatMode] = [
    LunaOpenAIResponses,
    LunaOpenAIChatCompletions,
  ]
  for mode in modes {
    let owner = @luna_event.LunaEventOwner::new(limits)
    owner.publish_accepted(openai_fixture_identity(), effective)
    let event = owner.view()
    let (actual, _) = openai_encode(
      openai_workspace(limits, 17),
      event,
      mode,
      42UL,
    )
    let expected = match mode {
      LunaOpenAIResponses =>
        b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_42\",\"object\":\"response\",\"model\":\"model-x\",\"status\":\"in_progress\"}}\n\n"
      LunaOpenAIChatCompletions =>
        b"data: {\"id\":\"resp_42\",\"object\":\"chat.completion.chunk\",\"model\":\"model-x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
    }
    openai_assert_bytes(actual, expected)
    owner.retire(event)
  }
}

///|
test "OpenAI Responses and Chat Token escaping goldens are canonical SSE" {
  let limits = openai_fixture_limits()
  let payload : FixedArray[Byte] = [
    b'x', b'"', b'\\', b'\x08', b'\x09', b'\x0a', b'\x0c', b'\x0d', b'\x01', 0xe4,
    0xb8, 0xad, 0xf0, 0x9f, 0x98, 0x80,
  ]
  let modes : FixedArray[LunaOpenAICompatMode] = [
    LunaOpenAIResponses,
    LunaOpenAIChatCompletions,
  ]
  for mode in modes {
    let owner = @luna_event.LunaEventOwner::new(limits)
    assert_eq(
      owner.publish_token_status(
        token_id=7,
        position=2,
        payload,
        source_offset=0,
        length=payload.length(),
      ),
      LunaEventPublishReady,
    )
    let event = owner.view()
    let (actual, _) = openai_encode(
      openai_workspace(limits, 1),
      event,
      mode,
      9UL,
    )
    let expected = match mode {
      LunaOpenAIResponses =>
        b"data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_9\",\"output_index\":0,\"content_index\":0,\"delta\":\"x\\\"\\\\\\b\\t\\n\\f\\r\\u0001\xe4\xb8\xad\xf0\x9f\x98\x80\"}\n\n"
      LunaOpenAIChatCompletions =>
        b"data: {\"id\":\"resp_9\",\"object\":\"chat.completion.chunk\",\"model\":\"model-x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\\\"\\\\\\b\\t\\n\\f\\r\\u0001\xe4\xb8\xad\xf0\x9f\x98\x80\"},\"finish_reason\":null}]}\n\n"
    }
    openai_assert_bytes(actual, expected)
    owner.retire(event)
  }
}

///|
test "OpenAI Responses and Chat Usage goldens are canonical SSE" {
  let limits = openai_fixture_limits()
  let modes : FixedArray[LunaOpenAICompatMode] = [
    LunaOpenAIResponses,
    LunaOpenAIChatCompletions,
  ]
  for mode in modes {
    let owner = @luna_event.LunaEventOwner::new(limits)
    owner.publish_usage(input_tokens=3, cached_input_tokens=1, output_tokens=2)
    let event = owner.view()
    let (actual, _) = openai_encode(
      openai_workspace(limits, 17),
      event,
      mode,
      5UL,
    )
    let expected = match mode {
      LunaOpenAIResponses =>
        b"data: {\"type\":\"response.usage\",\"response_id\":\"resp_5\",\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,\"output_tokens\":2,\"total_tokens\":5}}\n\n"
      LunaOpenAIChatCompletions =>
        b"data: {\"id\":\"resp_5\",\"object\":\"chat.completion.chunk\",\"model\":\"model-x\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"cached_tokens\":1,\"completion_tokens\":2,\"total_tokens\":5}}\n\n"
    }
    openai_assert_bytes(actual, expected)
    owner.retire(event)
  }
}

///|
test "OpenAI terminal Completed and Failed goldens append DONE exactly once" {
  let limits = openai_fixture_limits()
  let tail : FixedArray[Byte] = [b'!', b'\\']
  let code : FixedArray[Byte] = [
    b'c', b'a', b'p', b'a', b'c', b'i', b't', b'y', b'-', b'f', b'u', b'l', b'l',
  ]
  let modes : FixedArray[LunaOpenAICompatMode] = [
    LunaOpenAIResponses,
    LunaOpenAIChatCompletions,
  ]
  for mode in modes {
    let completed_owner = @luna_event.LunaEventOwner::new(limits)
    completed_owner.publish_completed(
      StopSequence,
      input_tokens=3,
      cached_input_tokens=1,
      output_tokens=2,
      tail,
      tail_offset=0,
      tail_length=tail.length(),
    )
    let completed_event = completed_owner.view()
    let (completed, _) = openai_encode(
      openai_workspace(limits, 17),
      completed_event,
      mode,
      8UL,
    )
    let completed_expected = match mode {
      LunaOpenAIResponses =>
        b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_8\",\"object\":\"response\",\"model\":\"model-x\",\"status\":\"completed\",\"finish_reason\":\"stop\",\"output_text\":\"!\\\\\",\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,\"output_tokens\":2,\"total_tokens\":5}}}\n\ndata: [DONE]\n\n"
      LunaOpenAIChatCompletions =>
        b"data: {\"id\":\"resp_8\",\"object\":\"chat.completion.chunk\",\"model\":\"model-x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\\\\\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"cached_tokens\":1,\"completion_tokens\":2,\"total_tokens\":5}}\n\ndata: [DONE]\n\n"
    }
    openai_assert_bytes(completed, completed_expected)
    completed_owner.retire(completed_event)

    let failed_owner = @luna_event.LunaEventOwner::new(limits)
    failed_owner.publish_failed(
      code,
      code_offset=0,
      code_length=code.length(),
      retryable=true,
    )
    let failed_event = failed_owner.view()
    let (failed, _) = openai_encode(
      openai_workspace(limits, 17),
      failed_event,
      mode,
      8UL,
    )
    let failed_expected = match mode {
      LunaOpenAIResponses =>
        b"data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_8\",\"status\":\"failed\",\"error\":{\"code\":\"capacity-full\",\"message\":\"request failed\",\"retryable\":true}}}\n\ndata: [DONE]\n\n"
      LunaOpenAIChatCompletions =>
        b"data: {\"id\":\"resp_8\",\"object\":\"chat.completion.chunk\",\"model\":\"model-x\",\"error\":{\"code\":\"capacity-full\",\"message\":\"request failed\",\"retryable\":true,\"type\":\"lunaflux_error\"}}\n\ndata: [DONE]\n\n"
    }
    openai_assert_bytes(failed, failed_expected)
    failed_owner.retire(failed_event)
  }
}
