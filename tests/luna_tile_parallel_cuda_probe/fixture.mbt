///|
fn luna_tile_probe_id(value : Int) -> @luna_tile_ir.LunaTileTensorId raise {
  @luna_tile_ir.LunaTileTensorId::from_stable_int(value)
}

///|
fn luna_tile_probe_tensor(
  id : Int,
  space : @luna_tile_ir.LunaTileAddressSpace,
  shape : ArrayView[Int64],
  strides : ArrayView[Int64],
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=luna_tile_probe_id(id),
    dtype=BFloat16,
    address_space=space,
    shape~,
    strides_bytes=strides,
    offset_bytes=0L,
    alignment_bytes=16L,
  )
}

///|
fn luna_tile_probe_region(
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
    tensor=luna_tile_probe_id(id),
    origins=[row, @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])],
    extents=[1L, 2L],
  )
}

///|
fn luna_tile_probe_program() -> @luna_tile_ir.LunaTileProgram raise {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[64L],
      tensors=[
        luna_tile_probe_tensor(1, Global, [64L, 2L], [4L, 2L]),
        luna_tile_probe_tensor(2, Shared, [1L, 2L], [4L, 2L]),
        luna_tile_probe_tensor(3, Register, [1L, 2L], [4L, 2L]),
        luna_tile_probe_tensor(4, Global, [64L, 2L], [4L, 2L]),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          luna_tile_probe_region(1, true),
          luna_tile_probe_region(2, false),
          stage,
        ),
        LunaTileBarrier(stage, Block),
        LunaTileCopy(
          luna_tile_probe_region(2, false),
          luna_tile_probe_region(3, false),
        ),
        LunaTileCopy(
          luna_tile_probe_region(3, false),
          luna_tile_probe_region(4, true),
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
}

///|
fn luna_tile_probe_candidates() -> (
  @luna_tile_ir.LunaTileCudaTranslationUnit,
  @luna_tile_ir.LunaTileParallelCandidate,
) raise {
  let program = luna_tile_probe_program()
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
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(1201),
    limits,
  )
  let parallel = @luna_tile_ir.specialize_luna_tile_parallel_candidate(
    program,
    planning,
    serial,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(1202),
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
fn luna_tile_probe_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn luna_tile_probe_fixture_record(
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  parallel : @luna_tile_ir.LunaTileParallelCandidate,
) -> Bytes {
  let mapping = parallel.mapping()
  @utf8.encode(
    "schema=lunaflux-lunatile-parallel-sm120-fixture.v1\n" +
    "program_sha256=" +
    serial.program_digest().as_hex() +
    "\nserial_translation_sha256=" +
    serial.digest().as_hex() +
    "\nserial_source_sha256=" +
    serial.source_digest().as_hex() +
    "\nserial_entry_point=" +
    serial.entry_point() +
    "\nserial_launch=1,1,1;1,1,1\nserial_shared_memory_bytes=" +
    serial.memory_plan().shared_memory_bytes().to_string() +
    "\nparallel_candidate_sha256=" +
    parallel.digest().as_hex() +
    "\nparallel_source_sha256=" +
    parallel.source_digest().as_hex() +
    "\nparallel_entry_point=" +
    parallel.entry_point() +
    "\nparallel_launch=" +
    mapping.block_count().to_string() +
    ",1,1;" +
    mapping.block_threads().to_string() +
    ",1,1\nparallel_shared_memory_bytes=" +
    parallel.shared_plan().total_shared_memory_bytes().to_string() +
    "\nscalar_oracle=exact-input-bytes\ninput_bytes=256\noutput_bytes=256\n" +
    "target=sm_120\nqualification_only=true\nmanifest_bindable=false\n" +
    "promotion_authority=absent\n",
  )
}
