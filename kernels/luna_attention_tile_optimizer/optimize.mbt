///|
priv struct AttentionRewritePassResult {
  regions : ReadOnlyArray[AttentionTileRewriteRegion]
  changed : Bool
}

///|
priv struct AttentionRewritePipelineState {
  regions : ReadOnlyArray[AttentionTileRewriteRegion]
  trace : ReadOnlyArray[AttentionTileRewriteStep]
}

///|
fn apply_rewrite_pass(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
  rule : AttentionTileRewriteRule,
  parallel_workgroup_target : Int32,
) -> AttentionRewritePassResult {
  match rule {
    FactorSharedKeyAcrossQueryMap =>
      apply_shared_key_query_map_pass(program, regions)
    HoistPagedKeyValuePageLookup =>
      apply_paged_key_value_page_lookup_pass(program, regions)
    FuseScoreTransform => apply_score_transform_pass(program, regions)
    FuseOnlineSoftmaxFold => apply_online_softmax_fold_pass(program, regions)
    ReuseOnlineSoftmaxTileStorage =>
      apply_online_softmax_tile_storage_pass(program, regions)
    PartitionOnlineSoftmaxFold =>
      apply_partition_online_softmax_pass(
        program, regions, parallel_workgroup_target,
      )
  }
}

///|
fn run_rewrite_pipeline(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  passes : ReadOnlyArray[AttentionTileRewriteRule],
  index : Int32,
  state : AttentionRewritePipelineState,
  parallel_workgroup_target : Int32,
) -> AttentionRewritePipelineState {
  if index == passes.length() {
    return state
  }
  let pass = passes[index]
  let result = apply_rewrite_pass(
    program,
    state.regions,
    pass,
    parallel_workgroup_target,
  )
  run_rewrite_pipeline(
    program,
    passes,
    index + 1,
    {
      regions: result.regions,
      trace: append_trace(state.trace, {
        rule: pass,
        changed: result.changed,
        region_count: result.regions.length(),
      }),
    },
    parallel_workgroup_target,
  )
}

///|
fn rewrite_rule_code(rule : AttentionTileRewriteRule) -> Int32 {
  match rule {
    HoistPagedKeyValuePageLookup => 1
    FuseScoreTransform => 2
    FuseOnlineSoftmaxFold => 3
    ReuseOnlineSoftmaxTileStorage => 4
    PartitionOnlineSoftmaxFold => 5
    FactorSharedKeyAcrossQueryMap => 6
  }
}

///|
fn reduction_topology_code(topology : AttentionTileReductionTopology) -> String {
  match topology {
    AttentionOrderedOnlineSoftmaxFold => "ordered:1"
    AttentionPartitionedOnlineSoftmaxFold(partitions) =>
      "partitioned:\{partitions}"
  }
}

///|
fn canonical_optimization(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
  trace : ReadOnlyArray[AttentionTileRewriteStep],
  reduction_topology : AttentionTileReductionTopology,
) -> Bytes {
  let builder = StringBuilder()
  builder <+ "luna-attention-tile-optimization-v2\n"
  builder <+ "program_sha256=\{program.digest().as_hex()}\n"
  builder <+ "reduction=\{reduction_topology_code(reduction_topology)}\n"
  for step in trace {
    builder <+
      "pass=\{rewrite_rule_code(step.rule)},\{step.changed},\{step.region_count}\n"
  }
  for region in regions {
    builder <+ "region=\{rewrite_rule_code(region.rule)}"
    for value_id in region.members {
      builder <+ ":\{value_id.as_int()}"
    }
    builder <+ ":root=\{region.root.as_int()}\n"
  }
  @utf8.encode(builder.to_string())
}

///|
/// Runs a bounded, deterministic pipeline of pure equational recognizers.
/// Each pass consumes and returns an immutable region set; no pass can observe
/// a clock, device, filesystem, global cache, or model-family name.
pub fn optimize_attention_tile_program(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  parallel_workgroup_target~ : Int32 = 32,
) -> AttentionTileOptimization raise AttentionTileOptimizerError {
  if parallel_workgroup_target <= 0 || parallel_workgroup_target > 1048576 {
    raise InvalidAttentionTileOptimization(InvalidRewritePipeline)
  }
  let passes = ReadOnlyArray::from_array([
    HoistPagedKeyValuePageLookup,
    FuseScoreTransform,
    FuseOnlineSoftmaxFold,
    ReuseOnlineSoftmaxTileStorage,
    PartitionOnlineSoftmaxFold,
    FactorSharedKeyAcrossQueryMap,
  ])
  let state = run_rewrite_pipeline(
    program,
    passes,
    0,
    {
      regions: ReadOnlyArray::from_array([]),
      trace: ReadOnlyArray::from_array([]),
    },
    parallel_workgroup_target,
  )
  if state.trace.length() != passes.length() || state.regions.length() == 0 {
    raise InvalidAttentionTileOptimization(InvalidRewritePipeline)
  }
  let partition_count = attention_partition_count(
    program, parallel_workgroup_target,
  )
  let reduction_topology = if partition_count == 1 {
    AttentionOrderedOnlineSoftmaxFold
  } else {
    AttentionPartitionedOnlineSoftmaxFold(partition_count)
  }
  let canonical = canonical_optimization(
    program,
    state.regions,
    state.trace,
    reduction_topology,
  )
  {
    program_digest: program.digest(),
    regions: state.regions,
    trace: state.trace,
    reduction_topology,
    canonical,
    digest: AttentionTileOptimizationDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
