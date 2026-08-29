///|
fn fused_benchmark_test_qkv_qualify(
  candidate : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
  protocol : FusedCandidateBenchmarkProtocol,
  observations : ArrayView[FusedCandidateBenchmarkPairedObservation],
) -> FusedCandidateBenchmarkQualificationEvidence raise FusedCandidateBenchmarkError {
  qualify_fused_qkv_candidate_benchmark(
    candidate,
    protocol,
    observations,
    maximum_evidence_bytes=1048576,
  )
}

///|
test "protocol rejects mismatched candidate widened shape set and count drift" {
  let candidate = fused_benchmark_test_set().qkv_candidate()
  let observations = fused_benchmark_test_observations(
    candidate.dispatch_canary(),
  )
  let exact_shape_set = fused_candidate_benchmark_shape_set_identity()
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      fused_benchmark_test_protocol(
        fused_benchmark_test_digest("e"),
        exact_shape_set,
        fused_candidate_benchmark_warmup_iterations(),
        fused_candidate_benchmark_trials_per_shape(),
        fused_candidate_benchmark_repetitions_per_arm(),
      ),
      observations,
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      CandidateIdentity
    ) => ()
    _ => fail("unexpected candidate-identity rejection")
  } noraise {
    _ => fail("mismatched candidate identity was admitted")
  }
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      fused_benchmark_test_protocol(
        candidate.candidate_digest(),
        fused_benchmark_test_digest("e"),
        fused_candidate_benchmark_warmup_iterations(),
        fused_candidate_benchmark_trials_per_shape(),
        fused_candidate_benchmark_repetitions_per_arm(),
      ),
      observations,
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      ShapeCoverage
    ) => ()
    _ => fail("unexpected widened-shape rejection")
  } noraise {
    _ => fail("widened shape-set identity was admitted")
  }
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      fused_benchmark_test_protocol(
        candidate.candidate_digest(),
        exact_shape_set,
        fused_candidate_benchmark_warmup_iterations() - 1,
        fused_candidate_benchmark_trials_per_shape(),
        fused_candidate_benchmark_repetitions_per_arm(),
      ),
      observations,
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      Protocol
    ) => ()
    _ => fail("unexpected warmup-count rejection")
  } noraise {
    _ => fail("warmup-count drift was admitted")
  }
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      fused_benchmark_test_protocol(
        candidate.candidate_digest(),
        exact_shape_set,
        fused_candidate_benchmark_warmup_iterations(),
        fused_candidate_benchmark_trials_per_shape() + 1,
        fused_candidate_benchmark_repetitions_per_arm(),
      ),
      observations,
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      Protocol
    ) => ()
    _ => fail("unexpected trial-count rejection")
  } noraise {
    _ => fail("trial-count drift was admitted")
  }
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      fused_benchmark_test_protocol(
        candidate.candidate_digest(),
        exact_shape_set,
        fused_candidate_benchmark_warmup_iterations(),
        fused_candidate_benchmark_trials_per_shape(),
        fused_candidate_benchmark_repetitions_per_arm() + 1,
      ),
      observations,
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      Protocol
    ) => ()
    _ => fail("unexpected repetition-count rejection")
  } noraise {
    _ => fail("repetition-count drift was admitted")
  }
}

///|
test "complete canonical trial order rejects reordering duplication and omission" {
  let candidate = fused_benchmark_test_set().qkv_candidate()
  let protocol = fused_benchmark_test_protocol(
    candidate.candidate_digest(),
    fused_candidate_benchmark_shape_set_identity(),
    fused_candidate_benchmark_warmup_iterations(),
    fused_candidate_benchmark_trials_per_shape(),
    fused_candidate_benchmark_repetitions_per_arm(),
  )
  let reordered = fused_benchmark_test_observations(candidate.dispatch_canary())
  let first = reordered[0]
  reordered[0] = reordered[1]
  reordered[1] = first
  try fused_benchmark_test_qkv_qualify(candidate, protocol, reordered) catch {
    MismatchedFusedCandidateBenchmark(ObservationOrder) => ()
    _ => fail("unexpected reordered-trial rejection")
  } noraise {
    _ => fail("reordered trials were admitted")
  }
  let duplicated = fused_benchmark_test_observations(
    candidate.dispatch_canary(),
  )
  duplicated[1] = duplicated[0]
  try fused_benchmark_test_qkv_qualify(candidate, protocol, duplicated) catch {
    MismatchedFusedCandidateBenchmark(ObservationOrder) => ()
    _ => fail("unexpected duplicate-trial rejection")
  } noraise {
    _ => fail("duplicate trials were admitted")
  }
  let missing = fused_benchmark_test_observations(candidate.dispatch_canary())
  ignore(missing.pop())
  try fused_benchmark_test_qkv_qualify(candidate, protocol, missing) catch {
    MismatchedFusedCandidateBenchmark(ShapeCoverage) => ()
    _ => fail("unexpected missing-trial rejection")
  } noraise {
    _ => fail("missing trial was admitted")
  }
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      protocol,
      fused_benchmark_test_observations(
        candidate.dispatch_canary(),
        wrong_pair_order=true,
      ),
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      ObservationOrder
    ) => ()
    _ => fail("unexpected pair-order rejection")
  } noraise {
    _ => fail("non-interleaved pair order was admitted")
  }
}

///|
test "zero work overflow canary imbalance and regressions fail closed" {
  try
    FusedCandidateBenchmarkArmObservation::new(
      duration_ns=1L,
      work_units=0L,
      resources_acquired=0L,
      resources_released=0L,
      live_resources_before=0L,
      live_resources_after=0L,
      live_bytes_before=0L,
      live_bytes_after=0L,
      resource_failures=0L,
      dispatch_canary_before=0,
      dispatch_canary_after=0,
    )
  catch {
    InvalidFusedCandidateBenchmark(WorkCounter) => ()
    _ => fail("unexpected zero-work rejection")
  } noraise {
    _ => fail("zero work was admitted")
  }
  let candidate = fused_benchmark_test_set().qkv_candidate()
  let protocol = fused_benchmark_test_protocol(
    candidate.candidate_digest(),
    fused_candidate_benchmark_shape_set_identity(),
    fused_candidate_benchmark_warmup_iterations(),
    fused_candidate_benchmark_trials_per_shape(),
    fused_candidate_benchmark_repetitions_per_arm(),
  )
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      protocol,
      fused_benchmark_test_observations(
        candidate.dispatch_canary(),
        corrupt_canary=true,
      ),
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      DispatchCanary
    ) => ()
    _ => fail("unexpected canary rejection")
  } noraise {
    _ => fail("failed dispatch canary was admitted")
  }
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      protocol,
      fused_benchmark_test_observations(
        candidate.dispatch_canary(),
        unbalanced_resources=true,
      ),
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      ResourceBalance
    ) => ()
    _ => fail("unexpected resource rejection")
  } noraise {
    _ => fail("resource imbalance was admitted")
  }
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      protocol,
      fused_benchmark_test_observations(
        candidate.dispatch_canary(),
        baseline_duration_ns=1000L,
        candidate_duration_ns=1001L,
      ),
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      WinPolicy
    ) => ()
    _ => fail("unexpected per-pair regression rejection")
  } noraise {
    _ => fail("candidate per-pair regression was admitted")
  }
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      protocol,
      fused_benchmark_test_observations(
        candidate.dispatch_canary(),
        baseline_duration_ns=1000L,
        candidate_duration_ns=960L,
      ),
    )
  catch {
    @luna_fused_candidate_qualification.MismatchedFusedCandidateBenchmark(
      WinPolicy
    ) => ()
    _ => fail("unexpected aggregate-win rejection")
  } noraise {
    _ => fail("sub-five-percent shape win was admitted")
  }
  try
    fused_benchmark_test_qkv_qualify(
      candidate,
      protocol,
      fused_benchmark_test_observations(
        candidate.dispatch_canary(),
        baseline_duration_ns=5000000000000000000L,
        candidate_duration_ns=4000000000000000000L,
      ),
    )
  catch {
    @luna_fused_candidate_qualification.OverflowedFusedCandidateBenchmark(
      DurationSummary
    ) => ()
    _ => fail("unexpected duration-overflow rejection")
  } noraise {
    _ => fail("overflowing duration summary was admitted")
  }
}
