///|
/// Bind an already selected backend policy to the generic column-fold plan.
/// The selected policy is explicit. Offline selection may
/// reduce the accumulator window without changing the ordered reduction.
fn qwen_ingress_column_plan(
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  policy~ : @resource_policy.Policy,
) -> @luna_projection_tile_compiler.ProjectionColumnFoldPlan raise FusedParallelAotError {
  guard compilation
    .schedule()
    .column_fold_plan(
      consumer_groups=policy.consumer_groups(),
      maximum_live_accumulators=policy.live_accumulators(),
    )
    is Some(plan) else {
    raise FusedParallelMismatch(CompilerPolicy)
  }
  plan
}

///|
fn qwen_default_ingress_policy(
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
) -> @resource_policy.Policy {
  let pipeline = compilation.schedule().matrix_pipeline().unwrap()
  qwen_ingress_policy(
    2,
    pipeline.stage_count(),
    pipeline.transfer_width() / pipeline.tile_width(),
    pipeline.fragment_stage_count(),
    rows=pipeline.row_width() / pipeline.tile_width(),
  )
}

///|
fn qwen_ingress_policy(
  live : Int,
  stages : Int,
  tiles : Int,
  fragments : Int,
  rows? : Int = 1,
) -> @resource_policy.Policy {
  @resource_policy.Policy::new(
    id="ingress-g4-a\{live}-s\{stages}-t\{tiles}-f\{fragments}" +
      (if rows == 1 { "" } else { "-r\{rows}" }),
    consumer_groups=4,
    live_accumulators=live,
    operand_stages=stages,
    transfer_tiles=tiles,
    fragment_stages=fragments,
    row_tiles=rows,
  ).unwrap()
}

///|
/// Supported geometry, not a timing model. Consumer count is fixed by this
/// CUDA kernel's 128-thread ABI; accumulator windows are independently legal.
pub fn qwen_ingress_policies(
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
) -> ReadOnlyArray[@resource_policy.Policy] {
  guard compilation.schedule().fuses_attention_ingress_epilogue() else {
    return []
  }
  guard compilation.schedule().matrix_pipeline() is Some(pipeline) else {
    return []
  }
  let policies = [qwen_default_ingress_policy(compilation)]
  let head = pipeline.output_tiles().to_int64() * 16L
  // These are this backend's realizable finite-ring choices, not universal IR
  // constants or asserted profitability thresholds. Measurements choose them.
  for rows in [1, 2, 4, 8] {
    for stages in 2..<=4 {
      for tiles in [1, 2, 4, 8, 16] {
        if pipeline.tile_count() % tiles != 0 ||
          pipeline.tile_count() / tiles < stages {
          continue
        }
        let row_width = rows.to_int64() * pipeline.tile_width().to_int64()
        let shared = (row_width + head) *
          pipeline.tile_width().to_int64() *
          tiles.to_int64() *
          stages.to_int64() *
          2L +
          row_width * head * 4L
        if shared > 49152L {
          continue
        }
        for fragments in 1..<=Int::min(2, tiles) {
          for live in 1..<=2 {
            let policy = qwen_ingress_policy(
              live,
              stages,
              tiles,
              fragments,
              rows~,
            )
            if !policies.contains(policy) {
              policies.push(policy)
            }
          }
        }
      }
    }
  }
  ReadOnlyArray::from_array(policies)
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::ingress_policy(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> @resource_policy.Policy {
  self.ingress_policy
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate::ingress_policy(
  self : FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
) -> @resource_policy.Policy {
  self.qualification.ingress_policy
}

///|
/// The schedule binds numeric, shape, layout and compiler choices; the emitted
/// source digest additionally invalidates measurements after lowering changes.
pub fn FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate::resource_scope(
  self : FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
  device~ : String,
  toolchain~ : String,
) -> @resource_policy.Scope? {
  let q = self.qualification
  let (input, output) = q.semantic.projection_shape()
  let (heads, kv, dimension) = q.semantic.attention_shape()
  @resource_policy.Scope::new(
    device~,
    toolchain~,
    shape="bf16:\{input}:\{output}:\{heads}:\{kv}:\{dimension}:\{q.profile.max_query_rows()}:\{q.profile.max_query_tokens()}",
    program=q.tile_compilation.digest().as_hex(),
  )
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate::resource_candidate(
  self : FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
) -> @resource_policy.Candidate {
  let q = self.qualification
  let pipeline = q.tile_compilation.schedule().matrix_pipeline().unwrap()
  let head = q.semantic.attention_shape().2.to_int64()
  let shared = (pipeline.row_width().to_int64() + head) *
    pipeline.transfer_width().to_int64() *
    pipeline.stage_count().to_int64() *
    2L +
    pipeline.row_width().to_int64() * head * 4L
  @resource_policy.Candidate::new(
    q.ingress_policy,
    artifact=self.source_digest.as_hex(),
    threads=FUSED_PARALLEL_THREADS,
    shared_bytes=shared,
    registers_per_thread=None,
    resident_groups=None,
  ).unwrap()
}

///|
/// Static shared storage needs no opt-in and the kernel ABI fixes the thread
/// count. Register/residency limits must be supplied with target observations;
/// this default imposes neither and does not claim a measured register count.
pub fn qwen_ingress_resource_budget() -> @resource_policy.ResourceBudget {
  @resource_policy.ResourceBudget::new(
    max_threads=FUSED_PARALLEL_THREADS,
    max_shared_bytes=49152L,
    max_registers_per_thread=None,
    min_resident_groups=None,
  ).unwrap()
}
