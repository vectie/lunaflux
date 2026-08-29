///|
async test "approved tiny BF16 inputs derive one exact complete candidate set" {
  let root = @fs.realpath("tests/reference_corpus")
  let inputs = load_approved_bf16_inputs(root)
  let corpus = approved_corpus_case(inputs.corpus)
  assert_eq(inputs.plan.operations().length(), 21)
  assert_eq(corpus.selected_ids.length(), 11)
  assert_eq(corpus.continuation, [1031, 2185, 688, 2844])
  let candidates = approved_candidate_set(inputs, "3".repeat(64), 13U, 1U, 115U)
  assert_eq(candidates.operation_count(), 21)
  assert_eq(candidates.model_identity(), inputs.plan.identity())
  assert_eq(candidates.target(), approved_target())
  assert_eq(candidates.profile(), approved_profile())
  assert_eq(approved_kv_layout(inputs.plan).tokens_per_page(), 8)
  assert_eq(approved_kv_layout(inputs.plan).total_page_count(), 32)
  assert_eq(candidates.profile().max_page_table_entries(), 32)
  assert_eq(approved_worker_limits().max_plan_pages(), 32)
  assert_eq(approved_worker_limits().max_sequence_tokens(), 256)
  assert_eq(approved_step_limits().max_pages(), 32)
  assert_eq(approved_step_limits().max_sequence_tokens(), 256)
  assert_eq(
    approved_sha256(candidates.candidate_set_bytes()),
    candidates.candidate_set_sha256(),
  )
  assert_eq(
    approved_sha256(candidates.inventory_bytes()),
    candidates.inventory_sha256(),
  )
}

///|
test "approved compiler version components are exact and bounded" {
  let policy = approved_compiler_policy("3".repeat(64), 13U, 1U, 115U)
  assert_eq(policy.identity().major(), 13)
  assert_eq(policy.identity().minor(), 1)
  assert_eq(policy.identity().patch(), 115)
  try approved_compiler_policy("3".repeat(64), 13U, 1U, 1000000U) catch {
    InvalidInput(_) => ()
    _ => fail("unexpected compiler-component rejection")
  } noraise {
    _ => fail("out-of-bounds compiler patch was admitted")
  }
}

///|
test "approved selected logits use a fixed predeclared BF16 tolerance" {
  let expected : FixedArray[Double] = [0.0, 0.1, -0.2]
  let actual : FixedArray[Double] = [0.0, 0.105, -0.209]
  assert_true(compare_approved_selected_logits(expected, actual) < 0.01)
  try compare_approved_selected_logits(expected, [0.0, 0.111, -0.2]) catch {
    NumericalMismatch(_) => ()
    _ => fail("unexpected selected-logit mismatch error")
  } noraise {
    _ => fail("out-of-tolerance selected logit was admitted")
  }
}

///|
test "compiled-set scalar validation rejects malformed identities" {
  assert_eq(
    compiled_value("schema=one\ntarget=sm_120\n", "target"),
    Some("sm_120"),
  )
  assert_eq(compiled_value("a=1\na=2\n", "a"), None)
  try require_compiled_sha256("A".repeat(64), "invalid") catch {
    InvalidCompiledSet("invalid") => ()
    _ => fail("unexpected digest validation error")
  } noraise {
    _ => fail("uppercase digest was admitted")
  }
}
