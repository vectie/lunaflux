///|
test "speculative generation retains accepted frontier with zero warmed allocation" {
  let (_, prompt, writer) = serial_wire_plan(7UL)
  writer.append_final_prefill_completed(prompt.prefill_row(0), 4)
  let completion = writer.submit()
  let limits = serial_limits()
  let inference = serial_inference_limits()
  let owners = FixedArray::makei(256, _ => {
    @serial_generation.SpeculativeGreedyGeneration::new(
      prompt,
      completion,
      limits,
      inference,
      maximum_new_tokens=8,
      maximum_context=16,
      stop_tokens=[],
    )
  })
  let drafts : FixedArray[Int] = [4, 5, 6]
  let targets : FixedArray[Int] = [5, 9, 0]
  let next_drafts : FixedArray[Int] = [9, 10, 11]
  let next_targets : FixedArray[Int] = [10, 11, 12]
  let last_drafts : FixedArray[Int] = [12, 13, 14]
  let last_targets : FixedArray[Int] = [13, 15]
  begin_allocations()
  for owner in owners {
    ignore(owner.next_verification(drafts, sequence_value=2UL))
    let rejected = @greedy_verification.verify_greedy_prefix(
      drafts,
      targets,
      remaining_new_tokens=7,
      vocabulary_size=32,
      stop_tokens=[],
    )
    owner.record_commit(rejected, targets[:2])
    let next = owner.next_verification(next_drafts, sequence_value=4UL)
    assert_eq(next.prefill_row(0).sequence_token_start(), 4)
    let accepted = @greedy_verification.verify_greedy_prefix(
      next_drafts,
      next_targets,
      remaining_new_tokens=5,
      vocabulary_size=32,
      stop_tokens=[],
    )
    owner.record_commit(accepted, next_targets)
    let last = owner.next_verification(last_drafts, sequence_value=5UL)
    assert_eq(last.token_count(), 2)
    let length = @greedy_verification.verify_greedy_prefix(
      last_drafts[:2],
      last_targets,
      remaining_new_tokens=2,
      vocabulary_size=32,
      stop_tokens=[],
    )
    owner.record_commit(length, last_targets)
    assert_eq(owner.token_count(), 8)
    assert_eq(owner.token_at(7), 15)
    assert_eq(
      owner.finish_reason(),
      @serial_generation.GenerationFinish::Length,
    )
  }
  assert_eq(end_allocations(), 0UL)
}
