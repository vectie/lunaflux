///|
fn equivalence_semantic_output(
  strings : Array[Bytes],
  special_policy : @tokenizer.DecodeSpecialPolicy,
) -> (@inference.LunaRequestSemanticLease, LunaIncrementalOutputLease) raise {
  let inference = fixture_inference()
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
  let semantic_work = write.finish()
  while semantic_work.progress().is_pending() {

  }
  let semantic_lease = semantic_work.take_lease()
  let workspace = LunaIncrementalOutputWorkspace::new(
    fixture_tokenizer(),
    inference,
    LunaIncrementalOutputStepBudget::new(1),
  )
  let output_work = workspace.begin_luna_request_semantics(
    semantic_lease.view(),
    special_policy~,
  )
  while output_work.progress() is LunaIncrementalOutputPending {

  }
  (semantic_lease, output_work.take_lease())
}

///|
fn assert_equivalent_status(
  legacy : IncrementalOutputStatus,
  semantic : IncrementalOutputStatus,
) -> Unit raise {
  assert_eq(legacy.is_ready(), semantic.is_ready())
  assert_eq(legacy.copied_bytes(), semantic.copied_bytes())
  assert_eq(legacy.stop_matched(), semantic.stop_matched())
}

///|
fn assert_equivalent_destination(
  legacy : FixedArray[Byte],
  semantic : FixedArray[Byte],
) -> Unit raise {
  assert_eq(legacy.length(), semantic.length())
  for index in 0..<legacy.length() {
    assert_eq(legacy[index], semantic[index])
  }
}

///|
fn assert_semantic_setup_equivalent(
  strings : Array[Bytes],
  tokens : Array[Int],
  special_policy : @tokenizer.DecodeSpecialPolicy,
  finish : Bool,
) -> Unit raise {
  let legacy = IncrementalOutput::new(
    fixture_tokenizer(),
    stops(strings),
    fixture_inference(),
    special_policy~,
  )
  let (semantic_authority, semantic) = equivalence_semantic_output(
    strings, special_policy,
  )
  // Successful output setup has copied every stop byte and no longer depends
  // on the semantic generation.
  semantic_authority.release()
  let legacy_destination = FixedArray::make(32, b'!')
  let semantic_destination = FixedArray::make(32, b'!')
  let mut stopped = false
  for token in tokens {
    let legacy_status = legacy.push_token_status(
      token,
      legacy_destination,
      destination_offset=2,
    )
    let semantic_status = semantic.push_token_status(
      token,
      semantic_destination,
      destination_offset=2,
    )
    assert_equivalent_status(legacy_status, semantic_status)
    assert_equivalent_destination(legacy_destination, semantic_destination)
    if legacy_status.stop_matched() {
      stopped = true
    }
  }
  if finish && !stopped {
    let legacy_status = legacy.finish_into_status(
      legacy_destination,
      destination_offset=2,
    )
    let semantic_status = semantic.finish_into_status(
      semantic_destination,
      destination_offset=2,
    )
    assert_equivalent_status(legacy_status, semantic_status)
    assert_equivalent_destination(legacy_destination, semantic_destination)
  }
  semantic.release()
}

///|
test "semantic-view setup preserves legacy incremental-output semantics" {
  // Longest match wins when two stop strings end at the same byte.
  assert_semantic_setup_equivalent([b"END", b"ND"], [256, 257], Preserve, false)
  // An unmatched stop prefix is retained and then flushed by finish.
  assert_semantic_setup_equivalent([b"END"], [258], Preserve, true)
  // Split UTF-8 and skipped special-token handling remain identical.
  assert_semantic_setup_equivalent([], [0xe4, 0xbd, 0xa0, 260], Skip, true)
  // Overlapping stop prefixes use the same terminal boundary.
  assert_semantic_setup_equivalent(
    [b"ABA", b"BAB"],
    [65, 66, 65],
    Preserve,
    false,
  )
}
