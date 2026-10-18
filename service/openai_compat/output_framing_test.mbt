///|
test "answer framing survives token splits, terminal tails and connection reuse" {
  let limits = openai_fixture_limits()
  let modes : Array[LunaOpenAICompatMode] = [
    LunaOpenAIChatCompletions,
    LunaOpenAIResponses,
  ]
  for streaming in [false, true] {
    for mode in modes {
      let workspace = openai_workspace(
        limits,
        1,
        buffered=true,
        output_framing=LunaOpenAIOutputFraming::answer_after(b"</think>"),
      )
      for sequence in [1UL, 2UL] {
        let owner = @luna_event.LunaEventOwner::new(limits)
        owner.publish_accepted(
          openai_fixture_identity(),
          @inference.EffectiveLimits::new(
            max_new_tokens=12,
            context_ceiling=32,
            limits,
          ),
        )
        let accepted = owner.view()
        ignore(openai_encode(workspace, accepted, mode, sequence, streaming~))
        owner.retire(accepted)
        let mut rendered = ""
        for
          position, text in [
            "private <", "/thi", "nk>你好", " </think>literal",
          ] {
          let delta = FixedArray::from_array(@utf8.encode(text).to_array())
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
          let (bytes, _) = openai_encode(
            workspace,
            event,
            mode,
            sequence,
            streaming~,
          )
          rendered += @utf8.decode(Bytes::from_array(bytes[:]))
          owner.retire(event)
        }
        owner.publish_completed(
          StopSequence,
          input_tokens=3,
          cached_input_tokens=0,
          output_tokens=4,
          [b'!'],
          tail_offset=0,
          tail_length=1,
        )
        let event = owner.view()
        let (bytes, _) = openai_encode(
          workspace,
          event,
          mode,
          sequence,
          streaming~,
        )
        rendered += @utf8.decode(Bytes::from_array(bytes[:]))
        owner.retire(event)
        assert_false(rendered.contains("private"))
        assert_true(rendered.contains("你好"))
        assert_true(rendered.contains(" </think>literal"))
        assert_true(rendered.contains("!"))
        assert_true(
          rendered.contains(
            if mode == LunaOpenAIChatCompletions {
              "\"completion_tokens\":4"
            } else {
              "\"output_tokens\":4"
            },
          ),
        )
        if !streaming {
          assert_true(rendered.contains("你好 </think>literal!"))
        }
      }
    }
  }
}

///|
test "framing handles overlapping markers and absent answer on token exhaustion" {
  let limits = openai_fixture_limits()
  let workspace = openai_workspace(
    limits,
    1,
    buffered=true,
    output_framing=LunaOpenAIOutputFraming::answer_after(b"aab"),
  )
  for sequence in [1UL, 2UL] {
    let owner = @luna_event.LunaEventOwner::new(limits)
    owner.publish_accepted(
      openai_fixture_identity(),
      @inference.EffectiveLimits::new(
        max_new_tokens=4,
        context_ceiling=16,
        limits,
      ),
    )
    let accepted = owner.view()
    ignore(
      openai_encode(
        workspace,
        accepted,
        LunaOpenAIChatCompletions,
        sequence,
        streaming=false,
      ),
    )
    owner.retire(accepted)
    let delta = FixedArray::from_array(b"private aa".to_array())
    ignore(
      owner.publish_token_status(
        token_id=7,
        position=0,
        delta,
        source_offset=0,
        length=delta.length(),
      ),
    )
    let token = owner.view()
    ignore(
      openai_encode(
        workspace,
        token,
        LunaOpenAIChatCompletions,
        sequence,
        streaming=false,
      ),
    )
    owner.retire(token)
    let tail = if sequence == 1UL { b"abanswer" } else { b"a" }
    owner.publish_completed(
      if sequence == 1UL {
        StopSequence
      } else {
        TokenLimit
      },
      input_tokens=3,
      cached_input_tokens=0,
      output_tokens=1,
      FixedArray::from_array(tail.to_array()),
      tail_offset=0,
      tail_length=tail.length(),
    )
    let event = owner.view()
    let (bytes, _) = openai_encode(
      workspace,
      event,
      LunaOpenAIChatCompletions,
      sequence,
      streaming=false,
    )
    let rendered = @utf8.decode(Bytes::from_array(bytes[:]))
    assert_false(rendered.contains("private"))
    assert_true(
      rendered.contains(
        if sequence == 1UL {
          "\"content\":\"answer\""
        } else {
          "\"content\":\"\""
        },
      ),
    )
    if sequence == 2UL {
      assert_true(rendered.contains("\"finish_reason\":\"length\""))
    }
    owner.retire(event)
  }
}
