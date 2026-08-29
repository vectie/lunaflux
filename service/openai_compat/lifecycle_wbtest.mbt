///|
fn openai_expect_rule(
  action : () -> Unit raise LunaOpenAICompatError,
  expected : LunaOpenAICompatRule,
) -> Unit raise {
  try action() catch {
    LunaOpenAICompatInvalid(rule) => assert_eq(rule, expected)
  } noraise {
    _ => fail("expected Luna OpenAI compatibility rejection")
  }
}

///|
fn openai_wb_fixture_limits() -> @inference.InferenceLimits raise {
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
    max_decoded_delta_bytes=32,
    max_deadline_millis=10000UL,
    max_top_k=256,
    max_temperature=8.0,
  )
}

///|
fn openai_wb_workspace(
  limits : @inference.InferenceLimits,
  budget : Int,
) -> LunaOpenAICompatWorkspace raise {
  let model : FixedArray[Byte] = [b'm', b'o', b'd', b'e', b'l', b'-', b'x']
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
fn openai_drive_ready(work : LunaOpenAICompatWork) -> Unit raise {
  while work.state() == LunaOpenAICompatPending {
    assert_eq(work.progress(), LunaOpenAICompatAdvanced)
    assert_true(work.last_work_units() > 0)
  }
  assert_eq(work.state(), LunaOpenAICompatReady)
}

///|
fn openai_token_event(
  limits : @inference.InferenceLimits,
) -> (@luna_event.LunaEventOwner, @luna_event.LunaEventView) raise {
  let owner = @luna_event.LunaEventOwner::new(limits)
  let payload : FixedArray[Byte] = [b'a', b'"', b'b']
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
  (owner, owner.view())
}

///|
test "Luna OpenAI compatibility reauthenticates every live phase" {
  let phases : FixedArray[Int] = [
    LUNA_OPENAI_COMPAT_SETUP,
    LUNA_OPENAI_COMPAT_CONFIGURE_PART,
    LUNA_OPENAI_COMPAT_LITERAL,
    LUNA_OPENAI_COMPAT_RAW,
    LUNA_OPENAI_COMPAT_LOAD_ESCAPE,
    LUNA_OPENAI_COMPAT_EMIT_ESCAPE,
    LUNA_OPENAI_COMPAT_DECIMAL_PREPARE,
    LUNA_OPENAI_COMPAT_DECIMAL_EMIT,
    LUNA_OPENAI_COMPAT_FINALIZING,
  ]
  for target in phases {
    let limits = openai_wb_fixture_limits()
    let (owner, event) = openai_token_event(limits)
    let workspace = openai_wb_workspace(limits, 1)
    let work = workspace.begin(
      event,
      LunaOpenAIChatCompletions,
      response_sequence=18446744073709551615UL,
    )
    while workspace.phase != target {
      assert_eq(work.state(), LunaOpenAICompatPending)
      assert_eq(work.progress(), LunaOpenAICompatAdvanced)
    }
    owner.retire(event)
    assert_eq(work.progress(), LunaOpenAICompatAdvanced)
    assert_eq(work.last_work_units(), 1)
    assert_eq(work.state(), LunaOpenAICompatFailed)
    assert_eq(
      work.failure(),
      LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent),
    )
    work.abort()
    openai_expect_rule(() => ignore(work.state()), LunaOpenAICompatEpoch)
  }
}

///|
test "Luna OpenAI compatibility Ready detaches semantic owner" {
  let limits = openai_wb_fixture_limits()
  let (owner, event) = openai_token_event(limits)
  let workspace = openai_wb_workspace(limits, 17)
  let work = workspace.begin(event, LunaOpenAIResponses, response_sequence=3UL)
  openai_drive_ready(work)
  owner.retire(event)
  let view = work.take_view()
  let destination : FixedArray[Byte] = FixedArray::make(view.length(), 0)
  assert_eq(
    view.copy_chunk_to(
      destination,
      destination_offset=0,
      source_offset=0,
      length=1,
    ),
    1,
  )
  view.release()
  openai_expect_rule(() => ignore(work.state()), LunaOpenAICompatEpoch)
}

///|
test "Luna OpenAI compatibility owner is Busy until abort or View release" {
  let limits = openai_wb_fixture_limits()
  let (owner1, event1) = openai_token_event(limits)
  let (owner2, event2) = openai_token_event(limits)
  let workspace = openai_wb_workspace(limits, 17)
  let aborted = workspace.begin(
    event1,
    LunaOpenAIResponses,
    response_sequence=1UL,
  )
  openai_expect_rule(
    () => {
      ignore(
        workspace.begin(event2, LunaOpenAIResponses, response_sequence=2UL),
      )
    },
    LunaOpenAICompatLifecycle,
  )
  aborted.abort()
  openai_expect_rule(() => ignore(aborted.progress()), LunaOpenAICompatEpoch)
  let work = workspace.begin(
    event2,
    LunaOpenAIChatCompletions,
    response_sequence=2UL,
  )
  openai_drive_ready(work)
  let view = work.take_view()
  openai_expect_rule(
    () => {
      ignore(
        workspace.begin(event1, LunaOpenAIResponses, response_sequence=3UL),
      )
    },
    LunaOpenAICompatLifecycle,
  )
  view.release()
  openai_expect_rule(() => ignore(view.length()), LunaOpenAICompatEpoch)
  owner1.retire(event1)
  owner2.retire(event2)
}

///|
test "Luna OpenAI compatibility stale View authentication precedes bounds" {
  let limits = openai_wb_fixture_limits()
  let (owner, event) = openai_token_event(limits)
  let workspace = openai_wb_workspace(limits, 1)
  let work = workspace.begin(event, LunaOpenAIResponses, response_sequence=1UL)
  openai_drive_ready(work)
  let view = work.take_view()
  view.release()
  let destination : FixedArray[Byte] = []
  openai_expect_rule(
    () => {
      ignore(
        view.copy_chunk_to(
          destination,
          destination_offset=-1,
          source_offset=-1,
          length=0,
        ),
      )
    },
    LunaOpenAICompatEpoch,
  )
  owner.retire(event)
}

///|
test "Luna OpenAI compatibility chunk copy is nonzero and budget bounded" {
  let limits = openai_wb_fixture_limits()
  let (owner, event) = openai_token_event(limits)
  let workspace = openai_wb_workspace(limits, 17)
  let work = workspace.begin(event, LunaOpenAIResponses, response_sequence=1UL)
  openai_drive_ready(work)
  let view = work.take_view()
  let destination : FixedArray[Byte] = FixedArray::make(view.length(), 0)
  for invalid in [0, 18] {
    openai_expect_rule(
      () => {
        ignore(
          view.copy_chunk_to(
            destination,
            destination_offset=0,
            source_offset=0,
            length=invalid,
          ),
        )
      },
      LunaOpenAICompatDestination,
    )
  }
  openai_expect_rule(
    () => {
      ignore(
        view.copy_chunk_to(
          destination,
          destination_offset=destination.length(),
          source_offset=view.length(),
          length=1,
        ),
      )
    },
    LunaOpenAICompatDestination,
  )
  view.release()
  owner.retire(event)
}

///|
test "Luna OpenAI compatibility epoch exhaustion is pre-mutation" {
  let limits = openai_wb_fixture_limits()
  let (owner, event) = openai_token_event(limits)
  let workspace = openai_wb_workspace(limits, 1)
  workspace.epoch = 0xffffffffffffffffUL
  openai_expect_rule(
    () => {
      ignore(workspace.begin(event, LunaOpenAIResponses, response_sequence=1UL))
    },
    LunaOpenAICompatEpoch,
  )
  assert_eq(workspace.phase, LUNA_OPENAI_COMPAT_IDLE)
  assert_eq(workspace.semantic_views.length(), 0)
  owner.retire(event)
}
