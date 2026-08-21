///|
fn wb_semantic_lease(
  inference : @inference.InferenceLimits,
  strings : Array[Bytes],
) -> @inference.LunaRequestSemanticLease raise {
  let storage = @inference.LunaRequestSemanticStorage::new(
    inference,
    @inference.LunaRequestSemanticStepBudget::new(1),
  )
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=strings.length(),
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  for value in strings {
    write.push_stop_string_length(value.length())
    for byte in value {
      write.push_stop_string_byte(byte)
    }
  }
  write.push_cache_scope_byte(b'x')
  let work = write.finish()
  let mut pending = true
  while pending {
    let progress = work.progress()
    if progress.is_failed() {
      fail("semantic fixture validation failed")
    }
    pending = progress.is_pending()
  }
  work.take_lease()
}

///|
fn wb_legacy_output_result(
  strings : Array[Bytes],
  tokens : Array[Int],
  special_policy : @tokenizer.DecodeSpecialPolicy,
) -> (Bytes, Bool) raise {
  let output = IncrementalOutput::new(
    fixture_tokenizer(),
    stops(strings),
    fixture_inference(),
    special_policy~,
  )
  let destination = FixedArray::make(128, b'!')
  let mut copied = 0
  let mut matched = false
  for token in tokens {
    if !matched {
      let status = output.push_token_status(
        token,
        destination,
        destination_offset=copied,
      )
      if !status.is_ready() {
        fail("legacy equivalence decode rejected")
      }
      copied += status.copied_bytes()
      matched = status.stop_matched()
    }
  }
  if !matched {
    let status = output.finish_into_status(
      destination,
      destination_offset=copied,
    )
    if !status.is_ready() {
      fail("legacy equivalence finish rejected")
    }
    copied += status.copied_bytes()
  }
  (Bytes::makei(copied, index => destination[index]), matched)
}

///|
fn wb_semantic_output_result(
  strings : Array[Bytes],
  tokens : Array[Int],
  special_policy : @tokenizer.DecodeSpecialPolicy,
) -> (Bytes, Bool) raise {
  let inference = fixture_inference()
  let semantic_lease = wb_semantic_lease(inference, strings)
  let workspace = LunaIncrementalOutputWorkspace::new(
    fixture_tokenizer(),
    inference,
    LunaIncrementalOutputStepBudget::new(1),
  )
  let work = workspace.begin_luna_request_semantics(
    semantic_lease.view(),
    special_policy~,
  )
  wb_drive_output_work(work)
  let output = work.take_lease()
  semantic_lease.release()
  let destination = FixedArray::make(128, b'!')
  let mut copied = 0
  let mut matched = false
  for token in tokens {
    if !matched {
      let status = output.push_token_status(
        token,
        destination,
        destination_offset=copied,
      )
      if !status.is_ready() {
        fail("semantic equivalence decode rejected")
      }
      copied += status.copied_bytes()
      matched = status.stop_matched()
    }
  }
  if !matched {
    let status = output.finish_into_status(
      destination,
      destination_offset=copied,
    )
    if !status.is_ready() {
      fail("semantic equivalence finish rejected")
    }
    copied += status.copied_bytes()
  }
  output.release()
  (Bytes::makei(copied, index => destination[index]), matched)
}

///|
test "legacy and semantic setup share the complete output engine" {
  let cases : Array[(Array[Bytes], Array[Int], @tokenizer.DecodeSpecialPolicy)] = [
    ([], [65, 66, 67], Preserve),
    ([b"ABABAC", b"BAB"], [65, 66, 65, 66, 65, 67], Preserve),
    ([b"END", b"xEND"], [259], Preserve),
    ([b"\xe4\xbd\xa0"], [0xe4, 0xbd, 0xa0], Preserve),
    ([], [260, 65], Preserve),
    ([], [260, 65], Skip),
  ]
  for item in cases {
    let (strings, tokens, special_policy) = item
    assert_eq(
      wb_semantic_output_result(strings, tokens, special_policy),
      wb_legacy_output_result(strings, tokens, special_policy),
    )
  }
}

///|
test "semantic setup copies one UTF-8 byte per charged text unit" {
  let inference = fixture_inference()
  let semantic_lease = wb_semantic_lease(inference, [b"ABABAC"])
  let workspace = LunaIncrementalOutputWorkspace::new(
    fixture_tokenizer(),
    inference,
    LunaIncrementalOutputStepBudget::new(1),
  )
  let work = workspace.begin_luna_request_semantics(
    semantic_lease.view(),
    special_policy=Preserve,
  )
  assert_eq(work.progress(), LunaIncrementalOutputPending)
  assert_eq(work.last_work_units(), 1)
  assert_eq(workspace.pattern_lengths[0], 0)
  for expected in 1..<=6 {
    assert_eq(work.progress(), LunaIncrementalOutputPending)
    assert_eq(work.last_work_units(), 1)
    assert_eq(workspace.pattern_lengths[0], expected)
  }
  wb_drive_output_work(work)
  let output = work.take_lease()
  assert_true(workspace.setup_semantics.is_empty())
  semantic_lease.release()
  let destination = FixedArray::make(8, b'!')
  for token in [65, 66, 65, 66, 65] {
    assert_true(
      output
      .push_token_status(token, destination, destination_offset=0)
      .is_ready(),
    )
  }
  let matched = output.push_token_status(67, destination, destination_offset=0)
  assert_true(matched.is_ready())
  assert_true(matched.stop_matched())
  output.release()
}

///|
test "semantic setup authenticates exact limits before workspace mutation" {
  let inference = fixture_inference()
  let foreign = @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=64,
    max_new_tokens=32,
    max_context_tokens=64,
    max_token_id=260,
    max_stop_token_ids=4,
    max_stop_strings=3,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=32,
    max_deadline_millis=1000UL,
    max_top_k=261,
    max_temperature=4.0,
  )
  let semantic_lease = wb_semantic_lease(foreign, [b"END"])
  let workspace = wb_output_workspace(1)
  let epoch = workspace.epoch
  try
    workspace.begin_luna_request_semantics(
      semantic_lease.view(),
      special_policy=Preserve,
    )
  catch {
    Invalid(rule=DecodeEnvelope, issue=Unsupported) => ()
    _ => fail("unexpected semantic-limit binding error")
  } noraise {
    _ => fail("foreign semantic limits were accepted")
  }
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_IDLE)
  assert_eq(workspace.epoch, epoch)
  assert_true(workspace.setup_semantics.is_empty())
  semantic_lease.release()
  let work = workspace.begin(stops([]), special_policy=Preserve)
  wb_drive_output_work(work)
  work.take_lease().release()

  let stale_lease = wb_semantic_lease(inference, [b"END"])
  let stale_view = stale_lease.view()
  stale_lease.release()
  let stale_epoch = workspace.epoch
  try
    workspace.begin_luna_request_semantics(stale_view, special_policy=Preserve)
  catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected stale semantic-view begin error")
  } noraise {
    _ => fail("stale semantic view was accepted")
  }
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_IDLE)
  assert_eq(workspace.epoch, stale_epoch)
}

///|
test "mid-setup stale semantic view remains failed and abortable" {
  let inference = fixture_inference()
  let semantic_lease = wb_semantic_lease(inference, [b"ABABAC"])
  let workspace = wb_output_workspace(1)
  let work = workspace.begin_luna_request_semantics(
    semantic_lease.view(),
    special_policy=Preserve,
  )
  assert_eq(work.progress(), LunaIncrementalOutputPending)
  semantic_lease.release()
  try work.progress() catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected mid-setup semantic stale error")
  } noraise {
    _ => fail("stale semantic setup progressed")
  }
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_FAILED)
  assert_true(workspace.setup_semantics.is_empty())
  assert_true(workspace.setup_stops is None)
  assert_eq(work.last_work_units(), 1)
  try work.progress() catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected replayed semantic stale error")
  } noraise {
    _ => fail("failed semantic setup stopped replaying its error")
  }
  assert_eq(work.last_work_units(), 0)
  try work.take_lease() catch {
    Invalid(rule=LunaIncrementalOutputWorkspace, issue=NotReady) => ()
    _ => fail("unexpected failed semantic lease error")
  } noraise {
    _ => fail("failed semantic setup published a lease")
  }
  try workspace.begin(stops([]), special_policy=Preserve) catch {
    Invalid(rule=LunaIncrementalOutputWorkspace, issue=Busy) => ()
    _ => fail("unexpected failed semantic busy error")
  } noraise {
    _ => fail("failed semantic work lost cleanup authority")
  }
  work.abort()
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_IDLE)
  let reuse = workspace.begin(stops([b"END"]), special_policy=Preserve)
  wb_drive_output_work(reuse)
  reuse.take_lease().release()
}

///|
test "semantic release before first byte fails with retained abort authority" {
  let semantic_lease = wb_semantic_lease(fixture_inference(), [b"END"])
  let workspace = wb_output_workspace(1)
  let work = workspace.begin_luna_request_semantics(
    semantic_lease.view(),
    special_policy=Preserve,
  )
  semantic_lease.release()
  assert_eq(work.progress(), LunaIncrementalOutputPending)
  try work.progress() catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected pre-copy semantic stale error")
  } noraise {
    _ => fail("released semantic view supplied its first byte")
  }
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_FAILED)
  work.abort()
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_IDLE)
}

///|
test "semantic release during failure-table setup cannot publish ready" {
  let semantic_lease = wb_semantic_lease(fixture_inference(), [b"ABABAC"])
  let workspace = wb_output_workspace(1)
  let work = workspace.begin_luna_request_semantics(
    semantic_lease.view(),
    special_policy=Preserve,
  )
  while workspace.phase != LUNA_INCREMENTAL_OUTPUT_SETUP_FAILURE {
    assert_eq(work.progress(), LunaIncrementalOutputPending)
  }
  semantic_lease.release()
  let mut failed = false
  while !failed {
    try work.progress() catch {
      Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => failed = true
      _ => fail("unexpected failure-table semantic stale error")
    } noraise {
      progress =>
        if progress is LunaIncrementalOutputReady {
          fail("stale semantic authority published ready output work")
        }
    }
  }
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_FAILED)
  try work.take_lease() catch {
    Invalid(rule=LunaIncrementalOutputWorkspace, issue=NotReady) => ()
    _ => fail("unexpected stale failure-table lease error")
  } noraise {
    _ => fail("stale failure-table work published a lease")
  }
  work.abort()
}

///|
test "semantic release after ready fails take and remains abortable" {
  let semantic_lease = wb_semantic_lease(fixture_inference(), [b"END"])
  let workspace = wb_output_workspace(17)
  let work = workspace.begin_luna_request_semantics(
    semantic_lease.view(),
    special_policy=Preserve,
  )
  wb_drive_output_work(work)
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_READY)
  semantic_lease.release()
  try work.take_lease() catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected ready-before-take semantic stale error")
  } noraise {
    _ => fail("stale ready semantic view published an output lease")
  }
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_FAILED)
  assert_true(workspace.setup_semantics.is_empty())
  try work.take_lease() catch {
    Invalid(rule=LunaIncrementalOutputWorkspace, issue=NotReady) => ()
    _ => fail("unexpected replayed failed take error")
  } noraise {
    _ => fail("failed ready semantic work published a lease")
  }
  work.abort()
  assert_eq(workspace.phase, LUNA_INCREMENTAL_OUTPUT_IDLE)
}
