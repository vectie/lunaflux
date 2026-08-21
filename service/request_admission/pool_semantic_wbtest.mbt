///|
test "semantic import preserves stops and external retention blocks claim release" {
  let now = Ref(11UL)
  let pool = fixture_preparation_pool(now, 1, 1, 8192UL)
  let work = pool
    .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
    .work()
  let mut prior_total = 0UL
  while work.state() is LunaRequestPreparationPending {
    ignore(pool.progress())
    assert_true(work.last_work_units() <= 1)
    assert_true(work.total_work_units() >= prior_total)
    prior_total = work.total_work_units()
  }
  assert_eq(work.state(), LunaRequestPreparationReady)
  let claim = work.take_prepared().take_claim()
  assert_true(claim.is_stop_token(9))
  assert_false(claim.is_stop_token(8))
  let external = @inference.LunaRequestStopTokenRetentionSlot::new()
  try! claim.stop_tokens.retain_into(external)
  try claim.release() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected retained semantic release error")
  } noraise {
    _ => fail("claim released while semantic stop authority was retained")
  }
  assert_true(external.is_live())
  assert_eq(claim.scheduler_request().request_id().value(), 7UL)
  assert_eq(pool.free_lane_count(), 0)
  try! external.release()
  claim.release()
  assert_eq(pool.free_lane_count(), 1)
  assert_false(external.is_live())
  assert_false(claim.is_stop_token(9))
}

///|
test "semantic storage int and byte cells are mandatory per preparation lane" {
  let tokenizer = fixture_tokenizer()
  let inference = fixture_inference()
  let input_capacity = Int::min(
    tokenizer.limits().max_input_bytes,
    inference.max_text_bytes(),
  )
  let output_capacity = Int::min(
    tokenizer.limits().max_output_tokens,
    inference.max_input_tokens(),
  )
  let exact_int = @tokenizer.LunaTokenizerWorker::required_int_cells(
      input_capacity,
    ) +
    @inference.LunaTokenBufferStorage::required_int_cells(output_capacity) +
    @inference.LunaRequestSemanticStorage::required_int_cells(inference) +
    @incremental_output.LunaIncrementalOutputWorkspace::required_int_cells(
      inference,
    ).to_uint64() +
    2UL
  let exact_byte = @tokenizer.LunaTokenizerWorker::required_byte_cells(
      input_capacity,
    ) +
    @inference.LunaRequestSemanticStorage::required_byte_cells(inference) +
    @incremental_output.LunaIncrementalOutputWorkspace::required_byte_cells(
      inference,
    ).to_uint64()
  for missing_int in [true, false] {
    try
      new_pool_with_clock(
        tokenizer,
        tokenizer.digest(),
        fixture_identity("0"),
        inference,
        () => 11UL,
        1,
        LunaRequestPreparationStepBudget::new(1),
        LunaRequestPreparationWorkLimit::new(8192UL),
        LunaRequestPreparationStorageBudget::new(
          int_cells=if missing_int { exact_int - 1UL } else { exact_int },
          byte_cells=if missing_int { exact_byte } else { exact_byte - 1UL },
          reference_cells=5UL,
        ),
      )
    catch {
      Invalid(LunaPreparationStorage) => ()
      _ => fail("unexpected semantic storage budget error")
    } noraise {
      _ => fail("undersized semantic storage budget was accepted")
    }
  }
  let exact = new_pool_with_clock(
    tokenizer,
    tokenizer.digest(),
    fixture_identity("0"),
    inference,
    () => 11UL,
    1,
    LunaRequestPreparationStepBudget::new(1),
    LunaRequestPreparationWorkLimit::new(8192UL),
    LunaRequestPreparationStorageBudget::new(
      int_cells=exact_int,
      byte_cells=exact_byte,
      reference_cells=5UL,
    ),
  )
  assert_eq(exact.free_lane_count(), 1)
}
