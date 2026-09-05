///|
fn cuda_compute_primitive(
  compute : @luna_attention_strategy.AttentionTileCompute,
  target : @catalog.DeviceTarget,
) -> CudaAttentionComputePrimitive raise CudaAttentionTileLoweringError {
  match compute {
    AttentionScalarDot => CudaScalarFma
    AttentionSubgroupDot => CudaSubgroupShuffleDot
    AttentionGroupedQuerySubgroupDot => CudaGroupedQuerySubgroupShuffleDot
    AttentionGroupedQuerySplitSubgroupDot =>
      CudaGroupedQuerySplitSubgroupShuffleDot
    AttentionMatrixDot =>
      if target.compute_major() >= 8 && target.supports_bf16() {
        CudaMatrixMultiplyAccumulate
      } else {
        raise InvalidCudaAttentionTileLowering(UnsupportedCompute)
      }
    AttentionMatrixQkPv =>
      if target.compute_major() >= 8 && target.supports_bf16() {
        CudaMatrixQkPvMultiplyAccumulate
      } else {
        raise InvalidCudaAttentionTileLowering(UnsupportedCompute)
      }
  }
}

///|
fn cuda_copy_primitive(
  memory : @luna_attention_strategy.AttentionTileMemory,
  target : @catalog.DeviceTarget,
) -> CudaAttentionCopyPrimitive raise CudaAttentionTileLoweringError {
  match memory {
    AttentionDirectGlobal => CudaDirectGlobalLoad
    AttentionStagedShared => CudaCooperativeSharedLoad
    AttentionAsyncStagedShared =>
      if target.compute_major() >= 8 {
        CudaAsynchronousSharedLoad
      } else {
        raise InvalidCudaAttentionTileLowering(UnsupportedCopy)
      }
  }
}

///|
fn cuda_compute_code(compute : CudaAttentionComputePrimitive) -> Int {
  match compute {
    CudaScalarFma => 1
    CudaSubgroupShuffleDot => 2
    CudaGroupedQuerySubgroupShuffleDot => 3
    CudaGroupedQuerySplitSubgroupShuffleDot => 6
    CudaMatrixMultiplyAccumulate => 4
    CudaMatrixQkPvMultiplyAccumulate => 5
  }
}

///|
fn cuda_copy_code(copy : CudaAttentionCopyPrimitive) -> Int {
  match copy {
    CudaDirectGlobalLoad => 1
    CudaCooperativeSharedLoad => 2
    CudaAsynchronousSharedLoad => 3
  }
}

///|
fn attention_phase_code(
  phase : @luna_attention_strategy.AttentionExecutionPhase,
) -> Int {
  match phase {
    Prefill => 1
    Decode => 2
    MixedCompatibility => 3
  }
}

///|
fn has_rewrite_rule(
  optimization : @luna_attention_tile_optimizer.AttentionTileOptimization,
  expected : @luna_attention_tile_optimizer.AttentionTileRewriteRule,
) -> Bool {
  for region in optimization.regions() {
    if region.rule() == expected {
      return true
    }
  }
  false
}

///|
/// Refines the generic functional schedule into CUDA launch and instruction
/// primitives. All vendor decisions are confined to this lowering boundary.
pub fn lower_attention_tile_schedule_to_cuda(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  optimization : @luna_attention_tile_optimizer.AttentionTileOptimization,
  schedule : @luna_attention_tile_schedule.AttentionTileSchedule,
  target : @catalog.DeviceTarget,
  maximum_shared_memory_bytes : Int64,
) -> CudaAttentionTileLowering raise CudaAttentionTileLoweringError {
  if target.compute_major() < 8 ||
    target.compute_major() > 12 ||
    target.compute_minor() < 0 ||
    target.compute_minor() > 9 ||
    maximum_shared_memory_bytes <= 0L {
    raise InvalidCudaAttentionTileLowering(UnsupportedTarget)
  }
  if optimization.program_digest() != program.digest() ||
    !has_rewrite_rule(optimization, FuseScoreTransform) ||
    !has_rewrite_rule(optimization, FuseOnlineSoftmaxFold) {
    raise InvalidCudaAttentionTileLowering(InvalidOptimization)
  }
  if schedule.program_digest() != program.digest() {
    raise InvalidCudaAttentionTileLowering(InvalidLaunch)
  }
  let key_value_partitions = schedule.key_value_partitions()
  if key_value_partitions <= 0 ||
    (
      key_value_partitions > 1 &&
      !has_rewrite_rule(optimization, PartitionOnlineSoftmaxFold)
    ) {
    raise InvalidCudaAttentionTileLowering(InvalidOptimization)
  }
  if schedule.reuses_online_softmax_tile_storage() &&
    !has_rewrite_rule(optimization, ReuseOnlineSoftmaxTileStorage) {
    raise InvalidCudaAttentionTileLowering(InvalidOptimization)
  }
  let problem = program.problem()
  let candidate = program.candidate()
  let (query_rows, _) = problem.dimensions()
  let (query_heads, _, head_dimension, _) = problem.shape().dimensions()
  let (query_tile, key_value_tile, tile_head) = candidate.tile().dimensions()
  let (subgroups, pipeline_stages, vector_bytes, shared_memory_bytes) = schedule.execution_geometry()
  let block_threads = subgroups * 32
  if tile_head != head_dimension ||
    block_threads <= 0 ||
    block_threads > 1024 ||
    block_threads % 32 != 0 {
    raise InvalidCudaAttentionTileLowering(InvalidLaunch)
  }
  if shared_memory_bytes > maximum_shared_memory_bytes {
    raise InvalidCudaAttentionTileLowering(SharedMemory)
  }
  // Ragged batches need at most one potential tile origin per query row. The
  // generated kernel maps the dense upper-bound ordinal to sequence-aligned
  // query tiles and returns surplus blocks. A later runtime worklist may
  // compact this without changing the generic schedule.
  let grid_x = match problem.phase() {
    Decode => problem.batch_rows()
    Prefill => query_rows
    MixedCompatibility => raise InvalidCudaAttentionTileLowering(InvalidLaunch)
  }
  let query_head_tile = schedule.loops()[1].geometry().1
  if query_head_tile <= 0 || query_heads % query_head_tile != 0 {
    raise InvalidCudaAttentionTileLowering(InvalidLaunch)
  }
  let grid_y = query_heads / query_head_tile
  if grid_x <= 0 || grid_x > 2147483647 || grid_y <= 0 || grid_y > 65535 {
    raise InvalidCudaAttentionTileLowering(InvalidLaunch)
  }
  let compute = cuda_compute_primitive(schedule.compute(), target)
  let copy = cuda_copy_primitive(schedule.memory(), target)
  let hoist_paged_key_value_page_lookup = schedule.hoists_paged_key_value_page_lookup()
  let reuse_online_softmax_tile_storage = schedule.reuses_online_softmax_tile_storage()
  let builder = StringBuilder()
  builder <+ "luna-cuda-attention-tile-lowering-v3\n"
  builder <+ "program_sha256=\{program.digest().as_hex()}\n"
  builder <+ "optimization_sha256=\{optimization.digest().as_hex()}\n"
  builder <+ "schedule_sha256=\{schedule.digest().as_hex()}\n"
  builder <+ "target=sm_\{target.compute_major()}\{target.compute_minor()}\n"
  builder <+ "phase=\{attention_phase_code(problem.phase())}\n"
  builder <+ "primitive=\{cuda_compute_code(compute)},\{cuda_copy_code(copy)}\n"
  builder <+
    "launch=\{grid_x},\{grid_y},\{key_value_partitions},\{block_threads},\{shared_memory_bytes}\n"
  builder <+ "tile=\{query_tile},\{key_value_tile},\{head_dimension}\n"
  builder <+ "pipeline=\{pipeline_stages},\{vector_bytes}\n"
  builder <+
    "hoist_paged_key_value_page_lookup=\{hoist_paged_key_value_page_lookup}\n"
  builder <+
    "reuse_online_softmax_tile_storage=\{reuse_online_softmax_tile_storage}\n"
  let canonical = @utf8.encode(builder.to_string())
  {
    program_digest: program.digest(),
    optimization_digest: optimization.digest(),
    schedule_digest: schedule.digest(),
    target,
    phase: problem.phase(),
    compute,
    copy,
    grid_x,
    grid_y,
    key_value_partitions,
    block_threads,
    query_tile,
    key_value_tile,
    head_dimension,
    pipeline_stages,
    vector_bytes,
    shared_memory_bytes,
    hoist_paged_key_value_page_lookup,
    reuse_online_softmax_tile_storage,
    matrix_storage_layout: schedule.matrix_storage_layout(),
    canonical,
    digest: CudaAttentionTileLoweringDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
