///|
test "speculative continuation follows accepted inputs and publishes only committed targets" {
  let initial = seed()
  let owner = @serial_generation.SpeculativeGreedyGeneration::new(
    initial,
    complete(initial, 4),
    limits(),
    inference(),
    maximum_new_tokens=8,
    maximum_context=16,
    stop_tokens=[],
  )
  let first = owner.next_verification([4, 5, 6], sequence_value=2UL)
  assert_eq(first.prefill_row(0).sequence_token_start(), 2)
  assert_eq(owner.token_count(), 1)
  let rejected = @greedy.verify_greedy_prefix(
    [4, 5, 6],
    [5, 9, 0],
    remaining_new_tokens=7,
    vocabulary_size=32,
    stop_tokens=[],
  )
  owner.record_commit(rejected, [5, 9])
  let next = owner.next_verification([9, 10, 11], sequence_value=4UL)
  assert_eq(next.prefill_row(0).sequence_token_start(), 4)
  assert_eq(next.page_count(), first.page_count())
  let accepted = @greedy.verify_greedy_prefix(
    [9, 10, 11],
    [10, 11, 12],
    remaining_new_tokens=5,
    vocabulary_size=32,
    stop_tokens=[],
  )
  owner.record_commit(accepted, [10, 11, 12])
  let last = owner.next_verification([12, 13, 14], sequence_value=5UL)
  assert_eq(last.prefill_row(0).sequence_token_start(), 7)
  assert_eq(last.token_count(), 2)
  let length = @greedy.verify_greedy_prefix(
    [12, 13],
    [13, 15],
    remaining_new_tokens=2,
    vocabulary_size=32,
    stop_tokens=[],
  )
  owner.record_commit(length, [13, 15])
  assert_eq(owner.finish_reason(), @serial_generation.GenerationFinish::Length)
  for i, token in [4, 5, 9, 10, 11, 12, 13, 15] {
    assert_eq(owner.token_at(i), token)
  }
}

///|
test "speculative cancellation and invalid seed do not publish another proposal" {
  let initial = seed()
  let owner = @serial_generation.SpeculativeGreedyGeneration::new(
    initial,
    complete(initial, 4),
    limits(),
    inference(),
    maximum_new_tokens=3,
    maximum_context=16,
    stop_tokens=[],
  )
  try owner.next_verification([7], sequence_value=2UL) catch {
    _ => ()
  } noraise {
    _ => fail("foreign emitted seed admitted")
  }
  ignore(owner.next_verification([4, 5], sequence_value=2UL))
  try owner.next_verification([4], sequence_value=3UL) catch {
    _ => ()
  } noraise {
    _ => fail("pending verification replaced")
  }
  owner.cancel()
  let prefix = @greedy.verify_greedy_prefix(
    [4, 5],
    [5, 6],
    remaining_new_tokens=2,
    vocabulary_size=32,
    stop_tokens=[],
  )
  owner.record_commit(prefix, [5, 6])
  assert_eq(owner.token_count(), 1)
  assert_eq(
    owner.finish_reason(),
    @serial_generation.GenerationFinish::Cancelled,
  )
}

///|
test "exact context limit clips the last speculative block before submission" {
  let initial = seed()
  let owner = @serial_generation.SpeculativeGreedyGeneration::new(
    initial,
    complete(initial, 4),
    limits(),
    inference(),
    maximum_new_tokens=14,
    maximum_context=16,
    stop_tokens=[],
  )
  for epoch = 2UL; owner.finish_reason() is Running; epoch = epoch + 1UL {
    let token = owner.token_at(owner.token_count() - 1)
    let proposed = [token, token + 1, token + 2]
    let remaining = owner.remaining_new_tokens()
    let plan = owner.next_verification(proposed, sequence_value=epoch)
    let rows = remaining.min(proposed.length())
    assert_eq(plan.token_count(), rows)
    assert_eq(
      plan.prefill_row(0).sequence_token_start(),
      owner.token_count() + 1,
    )
    assert_true(plan.prefill_row(0).sequence_token_start() + rows < 16)
    let targets = [token + 1, token + 2, token + 3]
    let prefix = @greedy.verify_greedy_prefix(
      proposed[:rows],
      targets[:rows],
      remaining_new_tokens=remaining,
      vocabulary_size=32,
      stop_tokens=[],
    )
    owner.record_commit(prefix, targets[:rows])
    assert_eq(
      plan.prefill_row(0).sequence_token_start() + rows,
      owner.token_count() + 1,
    )
  }
  assert_eq(owner.token_count(), 14)
  assert_eq(owner.remaining_new_tokens(), 0)
  assert_eq(owner.finish_reason(), @serial_generation.GenerationFinish::Length)
  // The final emitted token is not another KV input. Input plus output is 16,
  // while the last committed input ends at 15; no speculative row is padded
  // past this physical frontier and no extra continuation may be submitted.
  for i in 0..<14 {
    assert_eq(owner.token_at(i), 4 + i)
  }
  try owner.next_verification([17], sequence_value=7UL) catch {
    @serial_generation.SerialGenerationError::Finished => ()
    error => raise error
  } noraise {
    _ => fail("extra verification after exact-capacity generation")
  }
  try
    @serial_generation.SpeculativeGreedyGeneration::new(
      initial,
      complete(initial, 4),
      limits(),
      inference(),
      maximum_new_tokens=15,
      maximum_context=16,
      stop_tokens=[],
    )
  catch {
    @serial_generation.SerialGenerationError::InvalidLimit => ()
    error => raise error
  } noraise {
    _ => fail("one token beyond context capacity admitted")
  }
}
