///|
fn wb_output_workspace(budget : Int) -> LunaIncrementalOutputWorkspace raise {
  LunaIncrementalOutputWorkspace::new(
    fixture_tokenizer(),
    fixture_inference(),
    LunaIncrementalOutputStepBudget::new(budget),
  )
}

///|
fn wb_drive_output_work(work : LunaIncrementalOutputWork) -> Unit raise {
  let mut pending = true
  while pending {
    match work.progress() {
      LunaIncrementalOutputPending => pending = true
      LunaIncrementalOutputReady => pending = false
    }
  }
}

///|
test "Luna setup budget rejects zero and over-maximum work" {
  try LunaIncrementalOutputStepBudget::new(0) catch {
    Invalid(rule=LunaIncrementalOutputWorkspace, issue=Unsupported) => ()
    _ => fail("unexpected zero Luna setup budget error")
  } noraise {
    _ => fail("zero Luna setup budget was accepted")
  }
  try LunaIncrementalOutputStepBudget::new(65537) catch {
    Invalid(rule=LunaIncrementalOutputWorkspace, issue=Unsupported) => ()
    _ => fail("unexpected oversized Luna setup budget error")
  } noraise {
    _ => fail("oversized Luna setup budget was accepted")
  }
  assert_eq(LunaIncrementalOutputStepBudget::new(17).work_units(), 17)
}

///|
test "Luna workspace requirement queries match exact backing cells" {
  let inference = fixture_inference()
  let patterns = inference.max_stop_strings()
  let pattern_cells = patterns * inference.max_stop_string_bytes()
  assert_eq(
    LunaIncrementalOutputWorkspace::required_int_cells(inference),
    pattern_cells + patterns * 3,
  )
  assert_eq(
    LunaIncrementalOutputWorkspace::required_byte_cells(inference),
    pattern_cells +
    inference.max_stop_string_bytes() +
    3 +
    inference.max_decoded_delta_bytes(),
  )
  let workspace = LunaIncrementalOutputWorkspace::new(
    fixture_tokenizer(),
    inference,
    LunaIncrementalOutputStepBudget::new(1),
  )
  assert_eq(workspace.failure.length(), pattern_cells)
  assert_eq(workspace.pattern_lengths.length(), patterns)
  assert_eq(workspace.states.length(), patterns)
  assert_eq(workspace.next_states.length(), patterns)
  assert_eq(workspace.pattern_bytes.length(), pattern_cells)
  assert_eq(workspace.pending.length(), inference.max_stop_string_bytes() + 3)
  assert_eq(workspace.scratch.length(), inference.max_decoded_delta_bytes())
}

///|
test "legacy output sizes the shared engine to exact admitted stop shape" {
  let inference = fixture_inference()
  let empty = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([]),
    inference,
    special_policy=Preserve,
  )
  let empty_workspace = empty.lease.require_workspace()
  assert_eq(empty_workspace.pattern_bytes.length(), 0)
  assert_eq(empty_workspace.failure.length(), 0)
  assert_eq(empty_workspace.pattern_lengths.length(), 0)
  assert_eq(empty_workspace.pending.length(), 3)
  assert_eq(
    empty_workspace.scratch.length(),
    inference.max_decoded_delta_bytes(),
  )

  let shaped = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([b"END", b"\xe4\xbd\xa0"]),
    inference,
    special_policy=Preserve,
  )
  let shaped_workspace = shaped.lease.require_workspace()
  assert_eq(shaped_workspace.pattern_lengths.length(), 2)
  assert_eq(shaped_workspace.pattern_bytes.length(), 6)
  assert_eq(shaped_workspace.failure.length(), 6)
  assert_eq(shaped_workspace.states.length(), 2)
  assert_eq(shaped_workspace.next_states.length(), 2)
  assert_eq(shaped_workspace.pending.length(), 6)
}

///|
test "Luna setup budget charges every bounded transition" {
  let workspace = wb_output_workspace(1)
  let work = workspace.begin(
    stops([b"ABABAC", b"\xe4\xbd\xa0"]),
    special_policy=Preserve,
  )
  let mut calls = 0
  let mut pending = true
  while pending {
    let progress = work.progress()
    assert_true(work.last_work_units() <= 1)
    assert_true(work.last_work_units() >= 0)
    calls += 1
    pending = progress is LunaIncrementalOutputPending
  }
  assert_true(calls > 10)
  assert_eq(work.last_work_units(), 1)
  assert_eq(work.progress(), LunaIncrementalOutputReady)
  assert_eq(work.last_work_units(), 0)
  work.take_lease().release()
}

///|
test "Luna work is one-shot and retained aliases stale after lease transfer" {
  let workspace = wb_output_workspace(2)
  let work = workspace.begin(stops([b"END"]), special_policy=Preserve)
  try work.take_lease() catch {
    Invalid(rule=LunaIncrementalOutputWorkspace, issue=NotReady) => ()
    _ => fail("unexpected early Luna lease error")
  } noraise {
    _ => fail("unfinished Luna setup published a lease")
  }
  try workspace.begin(stops([]), special_policy=Skip) catch {
    Invalid(rule=LunaIncrementalOutputWorkspace, issue=Busy) => ()
    _ => fail("unexpected busy Luna workspace error")
  } noraise {
    _ => fail("busy Luna workspace accepted another setup")
  }
  wb_drive_output_work(work)
  let lease = work.take_lease()
  try work.progress() catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected transferred Luna work error")
  } noraise {
    _ => fail("transferred Luna work remained live")
  }
  try work.take_lease() catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected replayed Luna work error")
  } noraise {
    _ => fail("Luna work published its lease twice")
  }
  lease.release()
}

///|
test "released Luna aliases reject and workspace reuse advances epoch" {
  let workspace = wb_output_workspace(17)
  let first_work = workspace.begin(stops([b"END"]), special_policy=Preserve)
  wb_drive_output_work(first_work)
  let first = first_work.take_lease()
  let destination = FixedArray::make(16, b'!')
  assert_true(
    first.push_token_status(256, destination, destination_offset=0).is_ready(),
  )
  first.release()
  assert_false(
    first.push_token_status(257, destination, destination_offset=0).is_ready(),
  )
  assert_false(
    first.finish_into_status(destination, destination_offset=0).is_ready(),
  )
  try first.is_stopped() catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected released Luna query error")
  } noraise {
    _ => fail("released Luna lease retained query authority")
  }
  try first.release() catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected double Luna release error")
  } noraise {
    _ => fail("Luna lease released twice")
  }
  let second_work = workspace.begin(stops([]), special_policy=Skip)
  wb_drive_output_work(second_work)
  let second = second_work.take_lease()
  assert_true(
    second
    .push_token_without_stops_status(65, destination, destination_offset=2)
    .is_ready(),
  )
  assert_eq(destination[2], b'A')
  assert_false(
    first.push_token_status(65, destination, destination_offset=3).is_ready(),
  )
  second.release()
}

///|
test "aborted Luna setup frees workspace and stales retained work" {
  let workspace = wb_output_workspace(1)
  let work = workspace.begin(stops([b"ABABAC"]), special_policy=Preserve)
  assert_eq(work.progress(), LunaIncrementalOutputPending)
  work.abort()
  try work.abort() catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected double Luna abort error")
  } noraise {
    _ => fail("Luna work aborted twice")
  }
  let reused = workspace.begin(stops([b"END"]), special_policy=Preserve)
  wb_drive_output_work(reused)
  reused.take_lease().release()
}

///|
test "Luna lease preserves overlap and UTF-8 stop semantics" {
  let workspace = wb_output_workspace(17)
  let work = workspace.begin(stops([b"ABABAC"]), special_policy=Preserve)
  wb_drive_output_work(work)
  let lease = work.take_lease()
  let destination = FixedArray::make(16, b'!')
  for token in [65, 66, 65, 66, 65] {
    let prefix = lease.push_token_status(
      token,
      destination,
      destination_offset=0,
    )
    assert_true(prefix.is_ready())
    assert_false(prefix.stop_matched())
  }
  let overlap = lease.push_token_status(67, destination, destination_offset=0)
  assert_true(overlap.is_ready())
  assert_true(overlap.stop_matched())
  assert_eq(overlap.copied_bytes(), 0)
  lease.release()

  let utf8_work = workspace.begin(
    stops([b"\xe4\xbd\xa0"]),
    special_policy=Preserve,
  )
  wb_drive_output_work(utf8_work)
  let utf8_lease = utf8_work.take_lease()
  assert_true(
    utf8_lease
    .push_token_status(0xe4, destination, destination_offset=0)
    .is_ready(),
  )
  assert_true(
    utf8_lease
    .push_token_status(0xbd, destination, destination_offset=0)
    .is_ready(),
  )
  let matched = utf8_lease.push_token_status(
    0xa0,
    destination,
    destination_offset=0,
  )
  assert_true(matched.is_ready())
  assert_true(matched.stop_matched())
  assert_eq(matched.copied_bytes(), 0)
  assert_true(utf8_lease.is_stopped())
  utf8_lease.release()
}
