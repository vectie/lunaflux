///|
fn promotion_test_digest(
  character : Char,
) -> LunaBf16PromotionDigest raise LunaBf16PromotionError {
  LunaBf16PromotionDigest::from_sha256(String::make(64, character))
}

///|
fn promotion_profile_digest(
  character : Char,
) -> @luna_profile_priority.LunaKernelProfileDigest raise {
  @luna_profile_priority.LunaKernelProfileDigest::from_sha256(
    String::make(64, character),
  )
}

///|
fn promotion_profile_shape() -> @luna_profile_priority.LunaPagedProfileShape {
  try! @luna_profile_priority.LunaPagedProfileShape::new(
    tokens_per_page=2,
    prefill_row_count=1,
    decode_row_count=1,
    query_token_count=3,
    page_table_block_count=3,
    page_table_trace_digest=@luna_profile_priority.LunaPagedPageTableTraceDigest::from_sha256(
      "d".repeat(64),
    ),
    [
      @luna_profile_priority.LunaPagedProfileRow::new(
        phase=PagedPrefill,
        query_length=2,
        context_length=2,
        cache_position=0,
        page_table_start=0,
        block_count=1,
      ),
      @luna_profile_priority.LunaPagedProfileRow::new(
        phase=PagedDecode,
        query_length=1,
        context_length=3,
        cache_position=2,
        page_table_start=1,
        block_count=2,
      ),
    ],
  )
}

///|
fn promotion_profile(
  release : LunaBf16KernelReleaseInput,
  workload? : Char = 'a',
) -> @luna_profile_priority.LunaPagedKernelProfileCapture {
  let plan = try! producer_fixture_plan()
  let shape = promotion_profile_shape()
  let observations : Array[
    @luna_profile_priority.LunaPagedKernelProfileObservation,
  ] = []
  let mut total = 0L
  for index in 0..<plan.operations().length() {
    let self_time = (index + 1).to_int64()
    observations.push(
      try! @luna_profile_priority.LunaPagedKernelProfileObservation::new(
        operation_id=@plan.OperationId::new(index),
        shape~,
        call_count=1L,
        self_time_ns=self_time,
        inclusive_time_ns=self_time,
      ),
    )
    total += self_time
  }
  try! @luna_profile_priority.admit_luna_paged_kernel_profile(
    plan,
    release.contracts(),
    profile_id=producer_fixture_profile().id(),
    workload=promotion_profile_digest(workload),
    profiler=promotion_profile_digest('b'),
    trace=promotion_profile_digest('c'),
    sampled_wall_time_ns=1000L,
    declared_attributed_self_time_ns=total,
    observations,
    @luna_profile_priority.LunaKernelProfileLimits::new(
      maximum_observations=64,
      maximum_canonical_bytes=65536,
    ),
  )
}

///|
fn promotion_candidate_set(
  compiler? : @luna_specialization.LunaCompilerPolicy,
) -> LunaBf16CandidateSet {
  let plan = try! producer_fixture_plan()
  try! prepare_luna_bf16_candidate_set(
    plan,
    producer_fixture_layout(),
    producer_fixture_profile(),
    producer_fixture_target(),
    compiler.unwrap_or(try! producer_fixture_compiler()),
    try! LunaBf16CandidateSetLimits::new(
      max_operations=64,
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_total_bytes=33554432,
    ),
  )
}

///|
fn promotion_release(
  substituted_driver_operation? : Int = -1,
) -> LunaBf16KernelReleaseInput {
  let compilations = try! producer_fixture_compilations()
  if substituted_driver_operation >= 0 {
    let original = compilations[substituted_driver_operation]
    let module_digest = sha256_bytes(original.first_module)
    compilations[substituted_driver_operation] = LunaBf16OfflineCompilation::new(
      candidate=original.candidate,
      first_module=original.first_module,
      second_module=original.second_module,
      receipt=try! @luna_kernel_bundle.LunaDeterministicCompileReceipt::from_offline_builder(
        source_digest_sha256=original.candidate.source_sha256(),
        recipe_digest_sha256=original.candidate.recipe_sha256(),
        toolchain_digest_sha256="3".repeat(64),
        driver_identity_sha256="7".repeat(64),
        first_artifact_digest_sha256=module_digest,
        second_artifact_digest_sha256=module_digest,
      ),
    )
  }
  try! produce_luna_bf16_kernel_release(
    producer_fixture_plan(),
    producer_fixture_layout(),
    producer_fixture_target(),
    compilations,
    producer_fixture_limits(),
  )
}

///|
fn promotion_microbenchmarks(
  candidates : LunaBf16CandidateSet,
) -> Array[LunaBf16MicrobenchmarkObservation] {
  let result : Array[LunaBf16MicrobenchmarkObservation] = []
  let shape = try! LunaBf16BenchmarkShape::new(
    prefill_rows=1,
    decode_rows=1,
    query_tokens=3,
    page_table_entries=3,
    trace=promotion_test_digest('d'),
  )
  for candidate in candidates.candidates() {
    for trial in 0..<3 {
      result.push(
        try! LunaBf16MicrobenchmarkObservation::new(
          operation=candidate.operation_id(),
          shape_ordinal=0,
          trial_ordinal=trial,
          shape~,
          baseline_duration_ns=100L +
            candidate.operation_id().as_int().to_int64(),
          candidate_duration_ns=80L +
            candidate.operation_id().as_int().to_int64(),
          correctness=promotion_test_digest('e'),
          dispatch_canary=promotion_test_digest('f'),
          correctness_passed=true,
        ),
      )
    }
  }
  result
}

///|
fn promotion_end_to_end(
  workload? : Char = 'a',
) -> Array[LunaBf16EndToEndObservation] {
  let result : Array[LunaBf16EndToEndObservation] = []
  for trial in 0..<3 {
    result.push(
      try! LunaBf16EndToEndObservation::new(
        trial_ordinal=trial,
        workload=promotion_test_digest(workload),
        baseline_duration_ns=1000L,
        candidate_duration_ns=1000L,
        correctness=promotion_test_digest('9'),
        correctness_passed=true,
      ),
    )
  }
  result
}

///|
fn promotion_evidence_limits() -> LunaBf16ProfilerEvidenceLimits {
  try! LunaBf16ProfilerEvidenceLimits::new(
    maximum_candidates=64,
    maximum_shapes_per_candidate=4,
    trials_per_shape=3,
    end_to_end_trials=3,
    maximum_end_to_end_regression_ppm=0,
    maximum_canonical_bytes=1048576,
  )
}

///|
fn expect_promotion_rule(
  body : () -> Unit raise LunaBf16PromotionError,
  expected : LunaBf16PromotionRule,
) -> Unit raise {
  try body() catch {
    LunaBf16PromotionRejected(rule) => assert_eq(rule, expected)
  } noraise {
    _ => fail("hostile BF16 promotion evidence was admitted")
  }
}

///|
test "offline profiler evidence binds complete wins and release promotion" {
  let candidates = promotion_candidate_set()
  let release = promotion_release()
  let profile = promotion_profile(release)
  let evidence = try! collect_luna_bf16_profiler_evidence(
    candidates,
    profile,
    promotion_microbenchmarks(candidates),
    promotion_end_to_end(),
    promotion_evidence_limits(),
  )
  assert_eq(evidence.microbenchmark_count(), candidates.operation_count() * 3)
  assert_eq(evidence.end_to_end_count(), 3)
  let promotion = try! promote_luna_bf16_kernel_release(
    candidates,
    release,
    profile,
    evidence,
    try! LunaBf16PromotionLimits::new(maximum_canonical_bytes=1048576),
  )
  assert_eq(promotion.operation_count(), candidates.operation_count())
  assert_eq(promotion.profiler_evidence_digest(), evidence.digest())
  assert_eq(
    promotion.release_manifest_digest(),
    release.bundle().execution_manifest().digest_sha256(),
  )
  let repeated = try! collect_luna_bf16_profiler_evidence(
    candidates,
    profile,
    promotion_microbenchmarks(candidates),
    promotion_end_to_end(),
    promotion_evidence_limits(),
  )
  assert_eq(evidence.digest(), repeated.digest())
  assert_eq(evidence.canonical_bytes(), repeated.canonical_bytes())
}

///|
test "promotion rejects missing, slow, incorrect, and regressing evidence" {
  let candidates = promotion_candidate_set()
  let release = promotion_release()
  let profile = promotion_profile(release)
  let missing = promotion_microbenchmarks(candidates)
  ignore(missing.pop())
  expect_promotion_rule(
    () => {
      ignore(
        collect_luna_bf16_profiler_evidence(
          candidates,
          profile,
          missing,
          promotion_end_to_end(),
          promotion_evidence_limits(),
        ),
      )
    },
    CandidateCoverage,
  )
  let slow = promotion_microbenchmarks(candidates)
  slow[0] = {
    ..slow[0],
    candidate_duration_ns: slow[0].baseline_duration_ns + 1000L,
  }
  expect_promotion_rule(
    () => {
      ignore(
        collect_luna_bf16_profiler_evidence(
          candidates,
          profile,
          slow,
          promotion_end_to_end(),
          promotion_evidence_limits(),
        ),
      )
    },
    MicrobenchmarkWin,
  )
  let incorrect = promotion_microbenchmarks(candidates)
  incorrect[0] = { ..incorrect[0], correctness_passed: false }
  expect_promotion_rule(
    () => {
      ignore(
        collect_luna_bf16_profiler_evidence(
          candidates,
          profile,
          incorrect,
          promotion_end_to_end(),
          promotion_evidence_limits(),
        ),
      )
    },
    Correctness,
  )
  let regressing = promotion_end_to_end()
  regressing[1] = { ..regressing[1], candidate_duration_ns: 1001L }
  expect_promotion_rule(
    () => {
      ignore(
        collect_luna_bf16_profiler_evidence(
          candidates,
          profile,
          promotion_microbenchmarks(candidates),
          regressing,
          promotion_evidence_limits(),
        ),
      )
    },
    EndToEndRegression,
  )
}

///|
test "promotion rejects shape, observation-order, and canonical replay" {
  let candidates = promotion_candidate_set()
  let release = promotion_release()
  let profile = promotion_profile(release)
  let reordered = promotion_microbenchmarks(candidates)
  let first = reordered[0]
  reordered[0] = reordered[1]
  reordered[1] = first
  expect_promotion_rule(
    () => {
      ignore(
        collect_luna_bf16_profiler_evidence(
          candidates,
          profile,
          reordered,
          promotion_end_to_end(),
          promotion_evidence_limits(),
        ),
      )
    },
    ObservationOrder,
  )
  let oversized = promotion_microbenchmarks(candidates)
  oversized[0] = {
    ..oversized[0],
    shape: try! LunaBf16BenchmarkShape::new(
      prefill_rows=2,
      decode_rows=1,
      query_tokens=3,
      page_table_entries=3,
      trace=promotion_test_digest('d'),
    ),
  }
  expect_promotion_rule(
    () => {
      ignore(
        collect_luna_bf16_profiler_evidence(
          candidates,
          profile,
          oversized,
          promotion_end_to_end(),
          promotion_evidence_limits(),
        ),
      )
    },
    Shape,
  )
  let tiny_limit = try! LunaBf16ProfilerEvidenceLimits::new(
    maximum_candidates=64,
    maximum_shapes_per_candidate=4,
    trials_per_shape=3,
    end_to_end_trials=3,
    maximum_end_to_end_regression_ppm=0,
    maximum_canonical_bytes=4096,
  )
  expect_promotion_rule(
    () => {
      ignore(
        collect_luna_bf16_profiler_evidence(
          candidates,
          profile,
          promotion_microbenchmarks(candidates),
          promotion_end_to_end(),
          tiny_limit,
        ),
      )
    },
    CanonicalBytes,
  )
}

///|
test "promotion rejects profile and compiler replay across exact authority" {
  let candidates = promotion_candidate_set()
  let release = promotion_release()
  let profile = promotion_profile(release)
  let evidence = try! collect_luna_bf16_profiler_evidence(
    candidates,
    profile,
    promotion_microbenchmarks(candidates),
    promotion_end_to_end(),
    promotion_evidence_limits(),
  )
  let replayed_profile = promotion_profile(release, workload='7')
  expect_promotion_rule(
    () => {
      ignore(
        promote_luna_bf16_kernel_release(
          candidates,
          release,
          replayed_profile,
          evidence,
          try! LunaBf16PromotionLimits::new(maximum_canonical_bytes=1048576),
        ),
      )
    },
    ProfileCapture,
  )
  let alternate_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="8".repeat(64),
      major=13,
      minor=1,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let substituted = promotion_candidate_set(compiler=alternate_compiler)
  expect_promotion_rule(
    () => {
      ignore(
        promote_luna_bf16_kernel_release(
          substituted,
          release,
          profile,
          evidence,
          try! LunaBf16PromotionLimits::new(maximum_canonical_bytes=1048576),
        ),
      )
    },
    CandidateSet,
  )
  let substituted_release = promotion_release(substituted_driver_operation=1)
  expect_promotion_rule(
    () => {
      ignore(
        promote_luna_bf16_kernel_release(
          candidates,
          substituted_release,
          profile,
          evidence,
          try! LunaBf16PromotionLimits::new(maximum_canonical_bytes=1048576),
        ),
      )
    },
    Compiler,
  )
}
