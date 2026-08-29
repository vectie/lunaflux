///|
fn tensor_core_probe_id(value : Int) -> @luna_tile_ir.LunaTileTensorId raise {
  @luna_tile_ir.LunaTileTensorId::from_stable_int(value)
}

///|
fn tensor_core_probe_tensor(
  id : Int,
  dtype : @luna_tile_ir.LunaTileDType,
  space : @luna_tile_ir.LunaTileAddressSpace,
  shape : ArrayView[Int64],
  strides : ArrayView[Int64],
  offset? : Int64 = 0L,
  alignment? : Int64 = 32L,
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=tensor_core_probe_id(id),
    dtype~,
    address_space=space,
    shape~,
    strides_bytes=strides,
    offset_bytes=offset,
    alignment_bytes=alignment,
  )
}

///|
fn tensor_core_probe_region(
  id : Int,
  tiled : Bool,
) -> @luna_tile_ir.LunaTileRegion raise {
  let row = if tiled {
    @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[
      @luna_tile_ir.LunaTileAffineTerm::new(loop_dimension=0, coefficient=16L),
    ])
  } else {
    @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])
  }
  @luna_tile_ir.LunaTileRegion::new(
    tensor=tensor_core_probe_id(id),
    origins=[row, @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])],
    extents=[16L, 16L],
  )
}

///|
fn tensor_core_probe_program() -> @luna_tile_ir.LunaTileProgram raise {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  let rows = 1024L
  @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[64L],
      tensors=[
        tensor_core_probe_tensor(1, BFloat16, Global, [rows, 16L], [32L, 2L]),
        tensor_core_probe_tensor(
          2,
          BFloat16,
          Shared,
          [16L, 16L],
          [32L, 2L],
          alignment=16L,
        ),
        tensor_core_probe_tensor(3, BFloat16, Global, [rows, 16L], [32L, 2L]),
        tensor_core_probe_tensor(
          4,
          BFloat16,
          Shared,
          [16L, 16L],
          [32L, 2L],
          offset=512L,
          alignment=16L,
        ),
        tensor_core_probe_tensor(5, Float32, Global, [16L, 16L], [64L, 4L]),
        tensor_core_probe_tensor(6, Float32, Register, [16L, 16L], [64L, 4L]),
        tensor_core_probe_tensor(7, Float32, Register, [16L, 16L], [64L, 4L]),
        tensor_core_probe_tensor(8, Float32, Global, [rows, 16L], [64L, 4L]),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          tensor_core_probe_region(1, true),
          tensor_core_probe_region(2, false),
          stage,
        ),
        LunaTileAsyncCopy(
          tensor_core_probe_region(3, true),
          tensor_core_probe_region(4, false),
          stage,
        ),
        LunaTileBarrier(stage, Block),
        LunaTileCopy(
          tensor_core_probe_region(5, false),
          tensor_core_probe_region(6, false),
        ),
        LunaTileMma(
          destination=tensor_core_probe_id(7),
          left=tensor_core_probe_id(2),
          right=tensor_core_probe_id(4),
          accumulator=tensor_core_probe_id(6),
        ),
        LunaTileCopy(
          tensor_core_probe_region(7, false),
          tensor_core_probe_region(8, true),
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
fn tensor_core_probe_candidates() -> (
  @luna_tile_ir.LunaTileCudaTranslationUnit,
  @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Candidate,
) raise {
  let program = tensor_core_probe_program() catch {
    _ =>
      raise LunaTensorCoreProbeRejected("tensor-core fixture admission failed")
  }
  let planning = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=1024L,
    ),
  ) catch {
    _ => raise LunaTensorCoreProbeRejected("tensor-core serial planning failed")
  }
  let serial = @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program,
    planning,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(1401),
    @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=131072),
  ) catch {
    _ => raise LunaTensorCoreProbeRejected("tensor-core serial lowering failed")
  }
  let parallel = @luna_tile_ir.specialize_luna_tile_parallel_candidate(
    program,
    planning,
    serial,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(1402),
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
    _ =>
      raise LunaTensorCoreProbeRejected("tensor-core parallel planning failed")
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
  let candidate = @luna_tensor_core_candidate.lower_luna_tile_bf16_mma16x16x16_tensor_core_candidate(
    program,
    serial,
    parallel,
    contract,
    @luna_tensor_core_candidate.LunaTileTensorCoreCandidateLimits::new(
      maximum_source_bytes=131072,
      maximum_canonical_bytes=131072,
    ),
  ) catch {
    _ =>
      raise LunaTensorCoreProbeRejected("tensor-core candidate lowering failed")
  }
  (serial, candidate)
}
