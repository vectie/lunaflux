///|
test "first mismatch emits the target correction but does not commit its KV" {
  let prefix = @greedy_verification.verify_greedy_prefix(
    [10, 11, 12, 13],
    [11, 12, 7, 15],
    remaining_new_tokens=10,
    vocabulary_size=16,
    stop_tokens=[],
  )
  assert_eq(prefix.accepted_input_rows(), 3)
  assert_eq(prefix.accepted_draft_tokens(), 2)
  assert_eq(prefix.emitted_token_count(), 3)
  assert_eq(prefix.next_seed(), 7)
  assert_true(prefix.rollback_required())
  assert_eq(
    prefix.finish_reason(),
    @greedy_verification.GreedyVerificationFinish::Continue,
  )
}

///|
test "all accepted inputs keep state and emit a bonus token without consuming it" {
  let prefix = @greedy_verification.verify_greedy_prefix(
    [10, 11, 12],
    [11, 12, 15],
    remaining_new_tokens=10,
    vocabulary_size=16,
    stop_tokens=[],
  )
  assert_eq(prefix.accepted_input_rows(), 3)
  assert_eq(prefix.accepted_draft_tokens(), 2)
  assert_eq(prefix.next_seed(), 15)
  assert_false(prefix.rollback_required())
}

///|
test "stop and length boundaries force suffix restoration even when drafts match" {
  let inputs = [10, 11, 12, 13]
  let outputs = [11, 12, 13, 14]
  let stopped = @greedy_verification.verify_greedy_prefix(
    inputs,
    outputs,
    remaining_new_tokens=10,
    vocabulary_size=16,
    stop_tokens=[12],
  )
  assert_eq(stopped.accepted_input_rows(), 2)
  assert_eq(stopped.emitted_token_count(), 2)
  assert_true(stopped.rollback_required())
  assert_eq(
    stopped.finish_reason(),
    @greedy_verification.GreedyVerificationFinish::StopToken,
  )
  let limited = @greedy_verification.verify_greedy_prefix(
    inputs,
    outputs,
    remaining_new_tokens=2,
    vocabulary_size=16,
    stop_tokens=[],
  )
  assert_eq(limited.accepted_input_rows(), 2)
  assert_true(limited.rollback_required())
  assert_eq(
    limited.finish_reason(),
    @greedy_verification.GreedyVerificationFinish::Length,
  )
  let both = @greedy_verification.verify_greedy_prefix(
    inputs,
    outputs,
    remaining_new_tokens=2,
    vocabulary_size=16,
    stop_tokens=[12],
  )
  assert_eq(
    both.finish_reason(),
    @greedy_verification.GreedyVerificationFinish::StopToken,
  )
}

///|
test "every mismatch and truncation position reproduces the target sequence" {
  for rows in 1..<=6 {
    for mismatch in 0..<rows {
      let inputs = Array::makei(rows, row => 10 + row)
      let outputs = Array::makei(rows, row => {
        if row == mismatch {
          3
        } else {
          11 + row
        }
      })
      for remaining in 1..<=(rows + 1) {
        for stop_position in 0..<=rows {
          let stops = if stop_position == rows {
            []
          } else {
            [outputs[stop_position]]
          }
          let prefix = @greedy_verification.verify_greedy_prefix(
            inputs,
            outputs,
            remaining_new_tokens=remaining,
            vocabulary_size=32,
            stop_tokens=stops,
          )
          // Independent sequential target oracle: every reused input after the
          // seed must equal the preceding target output. Never consume a mismatch.
          let expected = (mismatch + 1)
            .min(rows)
            .min(remaining)
            .min(stop_position + 1)
          assert_eq(prefix.accepted_input_rows(), expected)
          assert_eq(prefix.emitted_token_count(), expected)
          assert_eq(prefix.next_seed(), outputs[expected - 1])
          assert_eq(prefix.rollback_required(), expected < rows)
          for row in 1..<expected {
            assert_eq(inputs[row], outputs[row - 1])
          }
        }
      }
    }
  }
}

///|
test "malformed or invalid later outputs never expose a partially accepted prefix" {
  for
    (inputs, outputs, remaining) in [
      ([], [], 1),
      ([1], [2, 3], 1),
      ([1], [2], 0),
    ] {
    try
      @greedy_verification.verify_greedy_prefix(
        inputs,
        outputs,
        remaining_new_tokens=remaining,
        vocabulary_size=16,
        stop_tokens=[],
      )
    catch {
      @greedy_verification.GreedyVerificationError::InvalidEnvelope => ()
      error => raise error
    } noraise {
      _ => fail("invalid verification envelope accepted")
    }
  }
  for
    (inputs, outputs, stops) in [
      ([1, 2], [2, 16], []),
      ([1, -1], [2, 3], []),
      ([1], [2], [16]),
    ] {
    try
      @greedy_verification.verify_greedy_prefix(
        inputs,
        outputs,
        remaining_new_tokens=1,
        vocabulary_size=16,
        stop_tokens=stops,
      )
    catch {
      @greedy_verification.GreedyVerificationError::InvalidToken => ()
      error => raise error
    } noraise {
      _ => fail("invalid later token ignored after early length stop")
    }
  }
}
