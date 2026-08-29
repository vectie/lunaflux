///|
fn tile_promotion_test_digest(
  character : Char,
) -> LunaTileOptimizerDigest raise {
  LunaTileOptimizerDigest::from_sha256(String::make(64, character))
}

///|
fn tile_promotion_test_id(value : Int) -> @luna_tile_ir.LunaTileTensorId raise {
  @luna_tile_ir.LunaTileTensorId::from_stable_int(value)
}

///|
fn tile_promotion_test_tensor(
  id : Int,
  space : @luna_tile_ir.LunaTileAddressSpace,
  shape : ArrayView[Int64],
  strides : ArrayView[Int64],
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=tile_promotion_test_id(id),
    dtype=BFloat16,
    address_space=space,
    shape~,
    strides_bytes=strides,
    offset_bytes=0L,
    alignment_bytes=16L,
  )
}

///|
fn tile_promotion_test_region(
  id : Int,
  tiled : Bool,
) -> @luna_tile_ir.LunaTileRegion raise {
  let row = if tiled {
    @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[
      @luna_tile_ir.LunaTileAffineTerm::new(loop_dimension=0, coefficient=1L),
    ])
  } else {
    @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])
  }
  @luna_tile_ir.LunaTileRegion::new(
    tensor=tile_promotion_test_id(id),
    origins=[row, @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])],
    extents=[1L, 2L],
  )
}

///|
fn tile_promotion_test_candidates(
  parallel_entry? : Int = 2202,
) -> (
  @luna_tile_ir.LunaTileCudaTranslationUnit,
  @luna_tile_ir.LunaTileParallelCandidate,
) raise {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  let program = @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[64L],
      tensors=[
        tile_promotion_test_tensor(1, Global, [64L, 2L], [4L, 2L]),
        tile_promotion_test_tensor(2, Shared, [1L, 2L], [4L, 2L]),
        tile_promotion_test_tensor(3, Register, [1L, 2L], [4L, 2L]),
        tile_promotion_test_tensor(4, Global, [64L, 2L], [4L, 2L]),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          tile_promotion_test_region(1, true),
          tile_promotion_test_region(2, false),
          stage,
        ),
        LunaTileBarrier(stage, Block),
        LunaTileCopy(
          tile_promotion_test_region(2, false),
          tile_promotion_test_region(3, false),
        ),
        LunaTileCopy(
          tile_promotion_test_region(3, false),
          tile_promotion_test_region(4, true),
        ),
      ],
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=4,
      max_instructions=5,
      max_rank=2,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=16384,
      max_shared_memory_bytes=4L,
    ),
  )
  let planning = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=4L,
    ),
  )
  let limits = @luna_tile_ir.LunaTileCudaLoweringLimits::new(
    max_source_bytes=65536,
  )
  let serial = @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program,
    planning,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(2201),
    limits,
  )
  let parallel = @luna_tile_ir.specialize_luna_tile_parallel_candidate(
    program,
    planning,
    serial,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(parallel_entry),
    @luna_tile_ir.LunaTileParallelPolicy::new(
      compute_major=12,
      compute_minor=0,
      block_threads=32,
      block_count=1,
      tile_m=1,
      tile_n=2,
      tile_k=1,
      pipeline_stages=2,
      vector_bytes=4,
      max_shared_memory_bytes=49152L,
      tensor_core_policy=ObserveEligibilityUseParallelSimt,
    ),
    limits,
  )
  (serial, parallel)
}

///|
fn tile_promotion_test_physical(
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  parallel : @luna_tile_ir.LunaTileParallelCandidate,
) -> AdmittedLunaTileSimtPhysicalEvidence raise {
  let serial_module = tile_promotion_test_digest('b')
  let module_digest = tile_promotion_test_digest('a')
  let fixture = tile_promotion_test_digest('f')
  let result = @utf8.encode(
    "outcome=lunatile-parallel-physical-campaign-pass\n" +
    "fixture_sha256=" +
    fixture.as_hex() +
    "\n" +
    "serial_source_sha256=" +
    serial.source_digest().as_hex() +
    "\n" +
    "parallel_source_sha256=" +
    parallel.source_digest().as_hex() +
    "\n" +
    "parallel_candidate_sha256=" +
    parallel.digest().as_hex() +
    "\n" +
    "serial_cubin_sha256=" +
    serial_module.as_hex() +
    "\n" +
    "parallel_cubin_sha256=" +
    module_digest.as_hex() +
    "\n" +
    "physical_cuda_observed=true\nmanifest_bindable=false\n" +
    "promotion_authority=absent\n",
  )
  let empty = LunaTileOptimizerDigest::from_sha256(
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  )
  let manifest = @utf8.encode(
    luna_tile_promotion_digest(result).as_hex() +
    "  CAMPAIGN_RESULT.txt\n" +
    module_digest.as_hex() +
    "  builds/parallel/first/kernel.cubin\n" +
    module_digest.as_hex() +
    "  builds/parallel/second/kernel.cubin\n" +
    serial_module.as_hex() +
    "  builds/serial/first/kernel.cubin\n" +
    serial_module.as_hex() +
    "  builds/serial/second/kernel.cubin\n" +
    fixture.as_hex() +
    "  fixture/fixture.v1\n" +
    parallel.digest().as_hex() +
    "  fixture/parallel.canonical\n" +
    parallel.source_digest().as_hex() +
    "  fixture/parallel.cu\n" +
    serial.digest().as_hex() +
    "  fixture/serial.canonical\n" +
    serial.source_digest().as_hex() +
    "  fixture/serial.cu\n" +
    empty.as_hex() +
    "  measurements/racecheck.stderr\n" +
    empty.as_hex() +
    "  measurements/runtime.stderr\n",
  )
  ignore(
    luna_tile_simt_result_fields(result) catch {
      error => fail("SIMT result fixture parse: \{Repr(error)}")
    },
  )
  ignore(
    luna_tile_simt_manifest_entries(manifest) catch {
      error => fail("SIMT manifest fixture parse: \{Repr(error)}")
    },
  )
  admit_luna_tile_simt_physical_evidence(
    parallel,
    serial,
    result,
    manifest,
    expected_files_manifest_digest=luna_tile_promotion_digest(manifest),
  ) catch {
    error => fail("SIMT physical fixture admission: \{Repr(error)}")
  }
}

///|
fn tile_promotion_test_benchmark(
  parallel : @luna_tile_ir.LunaTileParallelCandidate,
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  physical : AdmittedLunaTileSimtPhysicalEvidence,
) -> AdmittedLunaTilePairedBenchmarkWin raise {
  admit_luna_tile_paired_benchmark_win(
    path=LunaTileParallelSimtV1,
    candidate=LunaTileOptimizerDigest(parallel.digest().as_hex()),
    fallback=LunaTileOptimizerDigest(serial.digest().as_hex()),
    physical=physical.digest(),
    observations=[
      LunaTilePairedBenchmarkObservation::new(
        candidate_time_ns=800L,
        fallback_time_ns=1000L,
      ),
      LunaTilePairedBenchmarkObservation::new(
        candidate_time_ns=810L,
        fallback_time_ns=1010L,
      ),
      LunaTilePairedBenchmarkObservation::new(
        candidate_time_ns=790L,
        fallback_time_ns=990L,
      ),
    ],
    minimum_win_ppm=100_000,
  )
}

///|
fn tile_promotion_test_record() -> LunaTilePromotionApprovalRecord raise {
  let (serial, parallel) = tile_promotion_test_candidates()
  let physical = tile_promotion_test_physical(serial, parallel)
  prepare_luna_tile_simt_promotion_record(
    parallel,
    serial,
    physical,
    tile_promotion_test_benchmark(parallel, serial, physical),
    LunaTilePromotionArtifactInput::new(
      module_digest=physical.module_digest(),
      entry_point=parallel.entry_point(),
    ),
    deployment=LunaTilePromotionDeploymentIdentity::from_sha256("e".repeat(64)),
    promotion_epoch=7UL,
  )
}
