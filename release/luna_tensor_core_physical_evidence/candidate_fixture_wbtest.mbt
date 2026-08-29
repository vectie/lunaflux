///|
priv struct TensorCoreEvidenceCandidateFixture {
  program : @luna_tile_ir.LunaTileProgram
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit
  parallel : @luna_tile_ir.LunaTileParallelCandidate
  contract : @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Contract
  limits : @luna_tensor_core_candidate.LunaTileTensorCoreCandidateLimits
  candidate : @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Candidate
}

///|
fn tensor_core_evidence_id(value : Int) -> @luna_tile_ir.LunaTileTensorId raise {
  @luna_tile_ir.LunaTileTensorId::from_stable_int(value)
}

///|
fn tensor_core_evidence_tensor(
  id : Int,
  dtype : @luna_tile_ir.LunaTileDType,
  space : @luna_tile_ir.LunaTileAddressSpace,
  shape : ArrayView[Int64],
  strides : ArrayView[Int64],
  offset? : Int64 = 0L,
  alignment? : Int64 = 32L,
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=tensor_core_evidence_id(id),
    dtype~,
    address_space=space,
    shape~,
    strides_bytes=strides,
    offset_bytes=offset,
    alignment_bytes=alignment,
  )
}

///|
fn tensor_core_evidence_static_region(
  id : Int,
) -> @luna_tile_ir.LunaTileRegion raise {
  @luna_tile_ir.LunaTileRegion::new(
    tensor=tensor_core_evidence_id(id),
    origins=[
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
    ],
    extents=[16L, 16L],
  )
}

///|
fn tensor_core_evidence_tiled_region(
  id : Int,
) -> @luna_tile_ir.LunaTileRegion raise {
  @luna_tile_ir.LunaTileRegion::new(
    tensor=tensor_core_evidence_id(id),
    origins=[
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[
        @luna_tile_ir.LunaTileAffineTerm::new(loop_dimension=0, coefficient=16L),
      ]),
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
    ],
    extents=[16L, 16L],
  )
}

///|
fn tensor_core_evidence_program(
  tile_count? : Int64 = 64L,
) -> @luna_tile_ir.LunaTileProgram raise {
  let rows = tile_count * 16L
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[tile_count],
      tensors=[
        tensor_core_evidence_tensor(1, BFloat16, Global, [rows, 16L], [32L, 2L]),
        tensor_core_evidence_tensor(
          2,
          BFloat16,
          Shared,
          [16L, 16L],
          [32L, 2L],
          alignment=16L,
        ),
        tensor_core_evidence_tensor(3, BFloat16, Global, [rows, 16L], [32L, 2L]),
        tensor_core_evidence_tensor(
          4,
          BFloat16,
          Shared,
          [16L, 16L],
          [32L, 2L],
          offset=512L,
          alignment=16L,
        ),
        tensor_core_evidence_tensor(5, Float32, Global, [16L, 16L], [64L, 4L]),
        tensor_core_evidence_tensor(6, Float32, Register, [16L, 16L], [64L, 4L]),
        tensor_core_evidence_tensor(7, Float32, Register, [16L, 16L], [64L, 4L]),
        tensor_core_evidence_tensor(8, Float32, Global, [rows, 16L], [64L, 4L]),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          tensor_core_evidence_tiled_region(1),
          tensor_core_evidence_static_region(2),
          stage,
        ),
        LunaTileAsyncCopy(
          tensor_core_evidence_tiled_region(3),
          tensor_core_evidence_static_region(4),
          stage,
        ),
        LunaTileBarrier(stage, Block),
        LunaTileCopy(
          tensor_core_evidence_static_region(5),
          tensor_core_evidence_static_region(6),
        ),
        LunaTileMma(
          destination=tensor_core_evidence_id(7),
          left=tensor_core_evidence_id(2),
          right=tensor_core_evidence_id(4),
          accumulator=tensor_core_evidence_id(6),
        ),
        LunaTileCopy(
          tensor_core_evidence_static_region(7),
          tensor_core_evidence_tiled_region(8),
        ),
      ],
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=8,
      max_instructions=7,
      max_rank=2,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=32768,
      max_shared_memory_bytes=1024L,
    ),
  )
}

///|
fn tensor_core_evidence_candidate_fixture(
  tile_count? : Int64 = 64L,
) -> TensorCoreEvidenceCandidateFixture raise {
  let program = tensor_core_evidence_program(tile_count~) catch {
    error => fail("tensor-core evidence program: \{Repr(error)}")
  }
  let planning = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=1024L,
    ),
  )
  let serial = @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program,
    planning,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(1501),
    @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=131072),
  ) catch {
    error => fail("tensor-core evidence serial: \{Repr(error)}")
  }
  let parallel = @luna_tile_ir.specialize_luna_tile_parallel_candidate(
    program,
    planning,
    serial,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(1502),
    @luna_tile_ir.LunaTileParallelPolicy::new(
      compute_major=12,
      compute_minor=0,
      block_threads=32,
      block_count=1,
      tile_m=16,
      tile_n=16,
      tile_k=16,
      pipeline_stages=2,
      vector_bytes=4,
      max_shared_memory_bytes=65536L,
      tensor_core_policy=ObserveEligibilityUseParallelSimt,
    ),
    @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=131072),
  ) catch {
    error => fail("tensor-core evidence parallel: \{Repr(error)}")
  }
  let contract = @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Contract::new(
    compute_major=12,
    compute_minor=0,
    m=16,
    n=16,
    k=16,
    left_layout=TensorCoreRowMajor,
    right_layout=TensorCoreRowMajor,
    accumulator=TensorCoreAccumulatorF32,
    output_layout=TensorCoreRowMajor,
    left_alignment_bytes=32,
    right_alignment_bytes=32,
    accumulator_alignment_bytes=32,
    output_alignment_bytes=32,
    epilogue=TensorCoreIdentityEpilogue,
    block_threads=32,
    maximum_registers_per_thread=128,
  )
  let limits = @luna_tensor_core_candidate.LunaTileTensorCoreCandidateLimits::new(
    maximum_source_bytes=131072,
    maximum_canonical_bytes=131072,
  )
  let candidate = @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
    program, serial, parallel, contract, limits,
  )
  { program, serial, parallel, contract, limits, candidate }
}
