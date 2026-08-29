///|
test "SIMT promotion rejects candidate and artifact substitution" {
  let (serial, exact) = tile_promotion_test_candidates()
  let (_, substitute) = tile_promotion_test_candidates(parallel_entry=2203)
  let physical = tile_promotion_test_physical(serial, exact)
  let benchmark = tile_promotion_test_benchmark(exact, serial, physical)
  for candidate in [substitute] {
    try
      prepare_luna_tile_simt_promotion_record(
        candidate,
        serial,
        physical,
        benchmark,
        LunaTilePromotionArtifactInput::new(
          module_digest=physical.module_digest(),
          entry_point=candidate.entry_point(),
        ),
        deployment=LunaTilePromotionDeploymentIdentity::from_sha256(
          "e".repeat(64),
        ),
        promotion_epoch=7UL,
      )
    catch {
      LunaTileOptimizerPromotionMismatch(PhysicalEvidence) => ()
      _ => fail("candidate substitution returned the wrong refusal")
    } noraise {
      _ => fail("substituted candidate reached approval record")
    }
  }
  try
    prepare_luna_tile_simt_promotion_record(
      exact,
      serial,
      physical,
      benchmark,
      LunaTilePromotionArtifactInput::new(
        module_digest=tile_promotion_test_digest('f'),
        entry_point=exact.entry_point(),
      ),
      deployment=LunaTilePromotionDeploymentIdentity::from_sha256(
        "e".repeat(64),
      ),
      promotion_epoch=7UL,
    )
  catch {
    LunaTileOptimizerPromotionMismatch(PhysicalEvidence) => ()
    _ => fail("artifact substitution returned the wrong refusal")
  } noraise {
    _ => fail("substituted module reached approval record")
  }
}

///|
test "paired benchmark rejects non-win and fallback substitution" {
  let (serial, parallel) = tile_promotion_test_candidates()
  let physical = tile_promotion_test_physical(serial, parallel)
  try
    admit_luna_tile_paired_benchmark_win(
      path=LunaTileParallelSimtV1,
      candidate=LunaTileOptimizerDigest(parallel.digest().as_hex()),
      fallback=LunaTileOptimizerDigest(serial.digest().as_hex()),
      physical=physical.digest(),
      observations=[
        LunaTilePairedBenchmarkObservation::new(
          candidate_time_ns=1000L,
          fallback_time_ns=900L,
        ),
        LunaTilePairedBenchmarkObservation::new(
          candidate_time_ns=800L,
          fallback_time_ns=900L,
        ),
        LunaTilePairedBenchmarkObservation::new(
          candidate_time_ns=800L,
          fallback_time_ns=900L,
        ),
      ],
      minimum_win_ppm=1,
    )
  catch {
    LunaTileOptimizerPromotionMismatch(NonWin) => ()
    _ => fail("non-win returned the wrong refusal")
  } noraise {
    _ => fail("non-winning paired benchmark was admitted")
  }
  let exact = tile_promotion_test_benchmark(parallel, serial, physical)
  let substituted = admit_luna_tile_paired_benchmark_win(
    path=LunaTileParallelSimtV1,
    candidate=LunaTileOptimizerDigest(parallel.digest().as_hex()),
    fallback=tile_promotion_test_digest('f'),
    physical=physical.digest(),
    observations=[
      LunaTilePairedBenchmarkObservation::new(
        candidate_time_ns=800L,
        fallback_time_ns=1000L,
      ),
      LunaTilePairedBenchmarkObservation::new(
        candidate_time_ns=800L,
        fallback_time_ns=1000L,
      ),
      LunaTilePairedBenchmarkObservation::new(
        candidate_time_ns=800L,
        fallback_time_ns=1000L,
      ),
    ],
    minimum_win_ppm=100_000,
  )
  assert_true(exact.digest() != substituted.digest())
  try
    prepare_luna_tile_simt_promotion_record(
      parallel,
      serial,
      physical,
      substituted,
      LunaTilePromotionArtifactInput::new(
        module_digest=physical.module_digest(),
        entry_point=parallel.entry_point(),
      ),
      deployment=LunaTilePromotionDeploymentIdentity::from_sha256(
        "e".repeat(64),
      ),
      promotion_epoch=7UL,
    )
  catch {
    LunaTileOptimizerPromotionMismatch(Benchmark) => ()
    _ => fail("fallback substitution returned the wrong refusal")
  } noraise {
    _ => fail("substituted fallback reached approval record")
  }
}

///|
test "startup subject rejects replay and absence retains baseline" {
  let record = tile_promotion_test_record()
  let deployment = LunaTilePromotionDeploymentIdentity::from_sha256(
    "e".repeat(64),
  )
  luna_tile_require_startup_subject(
    record,
    deployment,
    7UL,
    record.candidate_digest(),
    record.fallback_digest(),
  )
  try
    luna_tile_require_startup_subject(
      record,
      deployment,
      8UL,
      record.candidate_digest(),
      record.fallback_digest(),
    )
  catch {
    LunaTileOptimizerPromotionMismatch(Replay) => ()
    _ => fail("stale epoch returned the wrong refusal")
  } noraise {
    _ => fail("approved record replayed across promotion epochs")
  }
  assert_true(
    select_luna_tile_optimizer_at_startup(
      None,
      deployment~,
      promotion_epoch=7UL,
      candidate=record.candidate_digest(),
      fallback=record.fallback_digest(),
    )
    is LunaTileBaselineSelected,
  )
}
