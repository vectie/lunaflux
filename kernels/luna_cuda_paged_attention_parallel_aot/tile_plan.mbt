///|
fn parallel_checked_product(
  left : Int,
  right : Int,
) -> Int raise PagedAttentionParallelAotError {
  if left <= 0 || right <= 0 || left > 2147483647 / right {
    raise PagedAttentionParallelOverflow(StaticResources)
  }
  left * right
}

///|
fn parallel_is_power_of_two(value : Int) -> Bool {
  value > 0 && (value & (value - 1)) == 0
}

///|
fn parallel_tensor(
  id : Int,
  dtype : @luna_tile_ir.LunaTileDType,
  space : @luna_tile_ir.LunaTileAddressSpace,
  extent : Int,
  offset : Int,
  alignment : Int,
) -> @luna_tile_ir.LunaTileTensorView raise PagedAttentionParallelAotError {
  @luna_tile_ir.LunaTileTensorView::new(
    id=@luna_tile_ir.LunaTileTensorId::from_stable_int(id),
    dtype~,
    address_space=space,
    shape=[extent.to_int64()],
    strides_bytes=[4L],
    offset_bytes=offset.to_int64(),
    alignment_bytes=alignment.to_int64(),
  ) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
}

///|
fn parallel_region(
  id : Int,
  extent : Int,
) -> @luna_tile_ir.LunaTileRegion raise PagedAttentionParallelAotError {
  @luna_tile_ir.LunaTileRegion::new(
    tensor=@luna_tile_ir.LunaTileTensorId::from_stable_int(id),
    origins=[@luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])],
    extents=[extent.to_int64()],
  ) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
}

///|
fn parallel_tile_aot(
  maximum_context_tokens : Int,
  shared_memory_bytes : Int,
  maximum_canonical_bytes : Int,
) -> (@luna_tile_ir.LunaTileProgram, @luna_tile_ir.LunaTileCudaAotInput) raise PagedAttentionParallelAotError {
  let threads = PAGED_ATTENTION_PARALLEL_BLOCK_THREADS
  let score_bytes = parallel_checked_product(maximum_context_tokens, 4)
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
  let global_partial = @luna_tile_ir.LunaTileTensorId::from_stable_int(1) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
  let shared_partial = @luna_tile_ir.LunaTileTensorId::from_stable_int(2) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
  let register_partial = @luna_tile_ir.LunaTileTensorId::from_stable_int(3) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
  let reduced = @luna_tile_ir.LunaTileTensorId::from_stable_int(4) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
  let score_scratch = @luna_tile_ir.LunaTileTensorId::from_stable_int(5) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
  let program = @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[1L],
      tensors=[
        parallel_tensor(1, Float32, Global, threads, 0, 16),
        parallel_tensor(2, Float32, Shared, threads, score_bytes, 16),
        parallel_tensor(3, Float32, Register, threads, 0, 4),
        parallel_tensor(4, Float32, Register, 1, 0, 4),
        parallel_tensor(5, Float32, Shared, maximum_context_tokens, 0, 16),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          parallel_region(1, threads),
          parallel_region(2, threads),
          stage,
        ),
        LunaTileBarrier(stage, Block),
        LunaTileCopy(parallel_region(2, threads), parallel_region(3, threads)),
        LunaTileReduce(
          Sum,
          destination=reduced,
          source=register_partial,
          axis=0,
        ),
        LunaTileReduce(
          Maximum,
          destination=reduced,
          source=register_partial,
          axis=0,
        ),
      ],
      constraints=[
        LunaTileDimensionEquals(
          @luna_tile_ir.LunaTileConstraintId::from_stable_int(1),
          shared_partial,
          0,
          threads.to_int64(),
        ),
        LunaTileAlignmentAtLeast(
          @luna_tile_ir.LunaTileConstraintId::from_stable_int(2),
          score_scratch,
          16L,
        ),
        LunaTileDimensionMultipleOf(
          @luna_tile_ir.LunaTileConstraintId::from_stable_int(3),
          global_partial,
          0,
          32L,
        ),
      ],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=5,
      max_instructions=6,
      max_rank=1,
      max_loop_rank=1,
      max_constraints=3,
      max_canonical_bytes=maximum_canonical_bytes,
      max_shared_memory_bytes=shared_memory_bytes.to_int64(),
    ),
  ) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
  let aot = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=16,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=shared_memory_bytes.to_int64(),
    ),
  ) catch {
    _ => raise PagedAttentionParallelInvalid(LunaTile)
  }
  if aot.memory_plan().shared_memory_bytes() != shared_memory_bytes.to_int64() ||
    aot.memory_plan().shared_tensor_count() != 2 {
    raise PagedAttentionParallelMismatch(StaticResources)
  }
  (program, aot)
}
