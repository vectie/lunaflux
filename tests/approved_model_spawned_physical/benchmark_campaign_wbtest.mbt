///|
test "physical benchmark canonical artifact storage is jointly bounded" {
  assert_true(physical_benchmark_artifact_lengths_valid(1, 1))
  assert_true(
    physical_benchmark_artifact_lengths_valid(
      PHYSICAL_BENCHMARK_MAX_CANONICAL_BYTES - 1,
      1,
    ),
  )
  assert_false(physical_benchmark_artifact_lengths_valid(0, 1))
  assert_false(physical_benchmark_artifact_lengths_valid(1, 0))
  assert_false(
    physical_benchmark_artifact_lengths_valid(
      PHYSICAL_BENCHMARK_MAX_CANONICAL_BYTES,
      1,
    ),
  )
  assert_false(
    physical_benchmark_artifact_lengths_valid(
      PHYSICAL_BENCHMARK_MAX_CANONICAL_BYTES + 1,
      1,
    ),
  )
}

///|
test "physical benchmark scope cannot imply comparison admission" {
  let scope = "correctness_scope=pinned-two-token-serving-contract\n" +
    "comparison_admission=not-run\ncleanup_complete=1\n"
  assert_true(scope.contains("comparison_admission=not-run"))
  assert_false(scope.contains("comparison_admission=pass"))
}
