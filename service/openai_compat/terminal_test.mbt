///|
fn openai_completed_expected(
  mode : LunaOpenAICompatMode,
  reason : @inference.FinishReason,
) -> Bytes {
  match (mode, reason) {
    (LunaOpenAIResponses, StopSequence) =>
      b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_6\",\"object\":\"response\",\"model\":\"model-x\",\"status\":\"completed\",\"finish_reason\":\"stop\",\"output_text\":\"\",\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,\"output_tokens\":2,\"total_tokens\":5}}}\n\ndata: [DONE]\n\n"
    (LunaOpenAIResponses, TokenLimit) =>
      b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_6\",\"object\":\"response\",\"model\":\"model-x\",\"status\":\"completed\",\"finish_reason\":\"length\",\"output_text\":\"\",\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,\"output_tokens\":2,\"total_tokens\":5}}}\n\ndata: [DONE]\n\n"
    (LunaOpenAIResponses, Cancelled) =>
      b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_6\",\"object\":\"response\",\"model\":\"model-x\",\"status\":\"completed\",\"finish_reason\":\"cancelled\",\"output_text\":\"\",\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,\"output_tokens\":2,\"total_tokens\":5}}}\n\ndata: [DONE]\n\n"
    (LunaOpenAIChatCompletions, StopSequence) =>
      b"data: {\"id\":\"resp_6\",\"object\":\"chat.completion.chunk\",\"model\":\"model-x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"cached_tokens\":1,\"completion_tokens\":2,\"total_tokens\":5}}\n\ndata: [DONE]\n\n"
    (LunaOpenAIChatCompletions, TokenLimit) =>
      b"data: {\"id\":\"resp_6\",\"object\":\"chat.completion.chunk\",\"model\":\"model-x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":3,\"cached_tokens\":1,\"completion_tokens\":2,\"total_tokens\":5}}\n\ndata: [DONE]\n\n"
    (LunaOpenAIChatCompletions, Cancelled) =>
      b"data: {\"id\":\"resp_6\",\"object\":\"chat.completion.chunk\",\"model\":\"model-x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"cancelled\"}],\"usage\":{\"prompt_tokens\":3,\"cached_tokens\":1,\"completion_tokens\":2,\"total_tokens\":5}}\n\ndata: [DONE]\n\n"
  }
}

///|
test "OpenAI terminal completion maps every finish reason exactly" {
  let limits = openai_fixture_limits()
  let empty : FixedArray[Byte] = []
  let modes : FixedArray[LunaOpenAICompatMode] = [
    LunaOpenAIResponses,
    LunaOpenAIChatCompletions,
  ]
  let reasons : FixedArray[@inference.FinishReason] = [
    StopSequence,
    TokenLimit,
    Cancelled,
  ]
  for mode in modes {
    for reason in reasons {
      let owner = @luna_event.LunaEventOwner::new(limits)
      owner.publish_completed(
        reason,
        input_tokens=3,
        cached_input_tokens=1,
        output_tokens=2,
        empty,
        tail_offset=0,
        tail_length=0,
      )
      let event = owner.view()
      let (actual, _) = openai_encode(
        openai_workspace(limits, 17),
        event,
        mode,
        6UL,
      )
      openai_assert_bytes(actual, openai_completed_expected(mode, reason))
      owner.retire(event)
    }
  }
}

///|
test "OpenAI failed retryable false maps exactly in both modes" {
  let limits = openai_fixture_limits()
  let code : FixedArray[Byte] = [
    b'n', b'o', b't', b'-', b'r', b'e', b't', b'r', b'y',
  ]
  let modes : FixedArray[LunaOpenAICompatMode] = [
    LunaOpenAIResponses,
    LunaOpenAIChatCompletions,
  ]
  for mode in modes {
    let owner = @luna_event.LunaEventOwner::new(limits)
    owner.publish_failed(
      code,
      code_offset=0,
      code_length=code.length(),
      retryable=false,
    )
    let event = owner.view()
    let (actual, _) = openai_encode(
      openai_workspace(limits, 17),
      event,
      mode,
      6UL,
    )
    let expected = match mode {
      LunaOpenAIResponses =>
        b"data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_6\",\"status\":\"failed\",\"error\":{\"code\":\"not-retry\",\"message\":\"request failed\",\"retryable\":false}}}\n\ndata: [DONE]\n\n"
      LunaOpenAIChatCompletions =>
        b"data: {\"id\":\"resp_6\",\"object\":\"chat.completion.chunk\",\"model\":\"model-x\",\"error\":{\"code\":\"not-retry\",\"message\":\"request failed\",\"retryable\":false,\"type\":\"lunaflux_error\"}}\n\ndata: [DONE]\n\n"
    }
    openai_assert_bytes(actual, expected)
    owner.retire(event)
  }
}
