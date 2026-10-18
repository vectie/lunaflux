///|
test "buffered response accumulates all visible deltas and terminal tail without SSE" {
  let limits = openai_fixture_limits()
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=4,
    context_ceiling=16,
    limits,
  )
  let modes : Array[LunaOpenAICompatMode] = [
    LunaOpenAIResponses,
    LunaOpenAIChatCompletions,
  ]
  for mode in modes {
    let workspace = openai_workspace(limits, 1, buffered=true)
    // Reusing the connection must not retain text from the previous response.
    for sequence in [1UL, 2UL] {
      let owner = @luna_event.LunaEventOwner::new(limits)
      owner.publish_accepted(openai_fixture_identity(), effective)
      let accepted = owner.view()
      assert_eq(
        openai_encode(workspace, accepted, mode, sequence, streaming=false).0.length(),
        0,
      )
      owner.retire(accepted)
      for position, bytes in [b"hello ", b"\"world\""] {
        let delta = FixedArray::from_array(bytes.to_array())
        assert_eq(
          owner.publish_token_status(
            token_id=7,
            position~,
            delta,
            source_offset=0,
            length=delta.length(),
          ),
          LunaEventPublishReady,
        )
        let event = owner.view()
        assert_eq(
          openai_encode(workspace, event, mode, sequence, streaming=false).0.length(),
          0,
        )
        owner.retire(event)
      }
      owner.publish_usage(
        input_tokens=3,
        cached_input_tokens=1,
        output_tokens=2,
      )
      let usage = owner.view()
      assert_eq(
        openai_encode(workspace, usage, mode, sequence, streaming=false).0.length(),
        0,
      )
      owner.retire(usage)
      owner.publish_completed(
        StopSequence,
        input_tokens=3,
        cached_input_tokens=1,
        output_tokens=2,
        [b'!'],
        tail_offset=0,
        tail_length=1,
      )
      let completed = owner.view()
      let (bytes, _) = openai_encode(
        workspace,
        completed,
        mode,
        sequence,
        streaming=false,
      )
      let expected = if mode == LunaOpenAIChatCompletions {
        "{\"id\":\"resp_\{sequence}\",\"object\":\"chat.completion\",\"model\":\"model-x\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"hello \\\"world\\\"!\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"cached_tokens\":1,\"completion_tokens\":2,\"total_tokens\":5}}"
      } else {
        "{\"id\":\"resp_\{sequence}\",\"object\":\"response\",\"model\":\"model-x\",\"status\":\"completed\",\"finish_reason\":\"stop\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello \\\"world\\\"!\"}]}],\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,\"output_tokens\":2,\"total_tokens\":5}}"
      }
      openai_assert_bytes(bytes, @utf8.encode(expected))
      owner.retire(completed)
    }
  }
}

///|
test "buffered storage is explicit and budget includes text and escaped output" {
  let limits = openai_fixture_limits()
  let streaming = LunaOpenAICompatWorkspace::required_byte_cells(limits)
  let buffered = LunaOpenAICompatWorkspace::required_byte_cells(
    limits,
    buffered=true,
  )
  let capacity = (limits.max_new_tokens() + 1) *
    limits.max_decoded_delta_bytes()
  assert_eq(
    buffered - streaming,
    (capacity * 7 - limits.max_decoded_delta_bytes() * 6).to_uint64(),
  )
  let owner = @luna_event.LunaEventOwner::new(limits)
  owner.publish_accepted(
    openai_fixture_identity(),
    @inference.EffectiveLimits::new(
      max_new_tokens=4,
      context_ceiling=16,
      limits,
    ),
  )
  let event = owner.view()
  let workspace = openai_workspace(limits, 1)
  try
    workspace.begin(
      event,
      LunaOpenAIChatCompletions,
      response_sequence=1UL,
      streaming=false,
    )
  catch {
    LunaOpenAICompatInvalid(LunaOpenAICompatCapacity) => ()
    e => raise e
  } noraise {
    _ => fail("unbudgeted buffered presentation accepted")
  }
  owner.retire(event)
}
