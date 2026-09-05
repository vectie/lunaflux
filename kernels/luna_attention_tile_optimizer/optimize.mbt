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
fn find_binding(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  id : @luna_attention_tile_ir.AttentionTileValueId,
) -> @luna_attention_tile_ir.AttentionTileBinding? {
  for binding in program.bindings() {
    if binding.id() == id {
      return Some(binding)
    }
  }
  None
}

///|
fn contains_region_root(
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
  rule : AttentionTileRewriteRule,
  root : @luna_attention_tile_ir.AttentionTileValueId,
) -> Bool {
  for region in regions {
    if region.rule == rule && region.root == root {
      return true
    }
  }
  false
}

///|
fn append_region(
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
  region : AttentionTileRewriteRegion,
) -> ReadOnlyArray[AttentionTileRewriteRegion] {
  let next : Array[AttentionTileRewriteRegion] = []
  for existing in regions {
    next.push(existing)
  }
  next.push(region)
  ReadOnlyArray::from_array(next)
}

///|
fn append_trace(
  trace : ReadOnlyArray[AttentionTileRewriteStep],
  step : AttentionTileRewriteStep,
) -> ReadOnlyArray[AttentionTileRewriteStep] {
  let next : Array[AttentionTileRewriteStep] = []
  for existing in trace {
    next.push(existing)
  }
  next.push(step)
  ReadOnlyArray::from_array(next)
}

///|
fn score_transform_members(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  score : @luna_attention_tile_ir.AttentionTileValueId,
) -> ReadOnlyArray[@luna_attention_tile_ir.AttentionTileValueId]? {
  let (scale_id, mask_id) = match find_binding(program, score) {
    Some(binding) =>
      match binding.expression() {
        AttentionTileCausalMask(scale) => (scale, Some(score))
        AttentionTileScale(_) => (score, None)
        _ => return None
      }
    None => return None
  }
  let dot_id = match find_binding(program, scale_id) {
    Some(binding) =>
      match binding.expression() {
        AttentionTileScale(dot) => dot
        _ => return None
      }
    None => return None
  }
  match find_binding(program, dot_id) {
    Some(binding) =>
      match binding.expression() {
        AttentionTileDot(_, _) =>
          match mask_id {
            Some(mask) =>
              Some(ReadOnlyArray::from_array([dot_id, scale_id, mask]))
            None => Some(ReadOnlyArray::from_array([dot_id, scale_id]))
          }
        _ => None
      }
    None => None
  }
}

///|
fn scan_score_transforms(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  index : Int,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
) -> ReadOnlyArray[AttentionTileRewriteRegion] {
  let bindings = program.bindings()
  if index == bindings.length() {
    return regions
  }
  let next = match bindings[index].expression() {
    AttentionTileOnlineSoftmax(scores~, values=_) =>
      match score_transform_members(program, scores) {
        Some(members) => {
          let root = members[members.length() - 1]
          if contains_region_root(regions, FuseScoreTransform, root) {
            regions
          } else {
            append_region(regions, { rule: FuseScoreTransform, members, root })
          }
        }
        None => regions
      }
    _ => regions
  }
  scan_score_transforms(program, index + 1, next)
}

///|
fn apply_score_transform_pass(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
) -> AttentionRewritePassResult {
  let next = scan_score_transforms(program, 0, regions)
  { regions: next, changed: next.length() != regions.length() }
}

///|
/// Page identity is invariant across every vector fragment of one paged K/V
/// row and is shared by the key and value loads. Recording this common region
/// lets a terminal backend compute the address once and distribute it with its
/// native subgroup primitive instead of repeating integer page arithmetic for
/// every vector fragment.
fn apply_paged_key_value_page_lookup_pass(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
) -> AttentionRewritePassResult {
  let mut key : @luna_attention_tile_ir.AttentionTileValueId? = None
  let mut value : @luna_attention_tile_ir.AttentionTileValueId? = None
  for binding in program.bindings() {
    match binding.expression() {
      AttentionTileInput(PagedKeyTileStream) => key = Some(binding.id())
      AttentionTileInput(PagedValueTileStream) => value = Some(binding.id())
      _ => ()
    }
  }
  match (key, value) {
    (Some(key), Some(value)) =>
      if contains_region_root(regions, HoistPagedKeyValuePageLookup, value) {
        { regions, changed: false }
      } else {
        {
          regions: append_region(regions, {
            rule: HoistPagedKeyValuePageLookup,
            members: ReadOnlyArray::from_array([key, value]),
            root: value,
          }),
          changed: true,
        }
      }
    _ => { regions, changed: false }
  }
}

///|
fn online_softmax_fold_members(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  normalize : @luna_attention_tile_ir.AttentionTileBinding,
) -> ReadOnlyArray[@luna_attention_tile_ir.AttentionTileValueId]? {
  let fold_id = match normalize.expression() {
    AttentionTileNormalize(fold) => fold
    _ => return None
  }
  let online_id = match find_binding(program, fold_id) {
    Some(binding) =>
      match binding.expression() {
        AttentionTileFoldKeyValueStream(online) => online
        _ => return None
      }
    None => return None
  }
  match find_binding(program, online_id) {
    Some(binding) =>
      match binding.expression() {
        AttentionTileOnlineSoftmax(scores=_, values=_) =>
          Some(ReadOnlyArray::from_array([online_id, fold_id, normalize.id()]))
        _ => None
      }
    None => None
  }
}

///|
fn scan_online_softmax_folds(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  index : Int,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
) -> ReadOnlyArray[AttentionTileRewriteRegion] {
  let bindings = program.bindings()
  if index == bindings.length() {
    return regions
  }
  let next = match online_softmax_fold_members(program, bindings[index]) {
    Some(members) => {
      let root = members[members.length() - 1]
      if contains_region_root(regions, FuseOnlineSoftmaxFold, root) {
        regions
      } else {
        append_region(regions, { rule: FuseOnlineSoftmaxFold, members, root })
      }
    }
    None => regions
  }
  scan_online_softmax_folds(program, index + 1, next)
}

///|
fn apply_online_softmax_fold_pass(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
) -> AttentionRewritePassResult {
  let next = scan_online_softmax_folds(program, 0, regions)
  { regions: next, changed: next.length() != regions.length() }
}

///|
/// Matrix QK consumes the staged key tile before online-softmax produces the
/// probability tile consumed by PV. Their lifetimes are disjoint. Likewise,
/// maximum and denominator are values of the query-local fold rather than
/// shared tensors. This pure region records those facts without choosing a
/// CUDA register, shared-memory address, or any other backend storage class.
fn apply_online_softmax_tile_storage_pass(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
) -> AttentionRewritePassResult {
  if program.candidate().compute() != AttentionMatrixQkPv {
    return { regions, changed: false }
  }
  for binding in program.bindings() {
    match online_softmax_fold_members(program, binding) {
      Some(members) => {
        let root = members[members.length() - 1]
        if contains_region_root(regions, ReuseOnlineSoftmaxTileStorage, root) {
          return { regions, changed: false }
        }
        return {
          regions: append_region(regions, {
            rule: ReuseOnlineSoftmaxTileStorage,
            members,
            root,
          }),
          changed: true,
        }
      }
      None => ()
    }
  }
  { regions, changed: false }
}

///|
fn attention_partition_count(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  parallel_workgroup_target : Int,
) -> Int {
  let problem = program.problem()
  let (query_rows, context_tokens) = problem.dimensions()
  let (query_tile, key_value_tile, _) = program.candidate().tile().dimensions()
  let (query_heads, key_value_heads, _, _) = problem.shape().dimensions()
  let query_tiles = 1 + (query_rows - 1) / query_tile
  let head_workgroups = match program.candidate().compute() {
    AttentionGroupedQuerySubgroupDot | AttentionGroupedQuerySplitSubgroupDot =>
      key_value_heads
    _ => query_heads
  }
  let natural_workgroups = query_tiles * problem.batch_rows() * head_workgroups
  // Partition only when the query/head map cannot provide one portable wave
  // of independent workgroups. This is a pure property of the semantic work
  // graph: a wider query tile or a larger continuous batch naturally removes
  // the need for duplicate KV traversal and a merge launch.
  let minimum_context = if problem.phase() == Decode { 128 } else { 512 }
  if problem.phase() == MixedCompatibility ||
    context_tokens < minimum_context ||
    natural_workgroups >= parallel_workgroup_target {
    return 1
  }
  let key_value_tiles = 1 + (context_tokens - 1) / key_value_tile
  let context_partitions = Int::min(
    8,
    Int::max(2, 1 + (context_tokens - 1) / (minimum_context / 2)),
  )
  let saturation_partitions = 1 +
    (parallel_workgroup_target - 1) / natural_workgroups
  Int::max(
    1,
    Int::min(
      context_partitions,
      Int::min(saturation_partitions, key_value_tiles),
    ),
  )
}

///|
fn apply_partition_online_softmax_pass(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
  parallel_workgroup_target : Int,
) -> AttentionRewritePassResult {
  if attention_partition_count(program, parallel_workgroup_target) == 1 {
    return { regions, changed: false }
  }
  let bindings = program.bindings()
  for binding in bindings {
    match online_softmax_fold_members(program, binding) {
      Some(members) => {
        let root = members[members.length() - 1]
        if contains_region_root(regions, PartitionOnlineSoftmaxFold, root) {
          return { regions, changed: false }
        }
        let next = append_region(regions, {
          rule: PartitionOnlineSoftmaxFold,
          members,
          root,
        })
        return { regions: next, changed: true }
      }
      None => ()
    }
  }
  { regions, changed: false }
}

///|
fn apply_rewrite_pass(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
  rule : AttentionTileRewriteRule,
  parallel_workgroup_target : Int,
) -> AttentionRewritePassResult {
  match rule {
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
  index : Int,
  state : AttentionRewritePipelineState,
  parallel_workgroup_target : Int,
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
fn rewrite_rule_code(rule : AttentionTileRewriteRule) -> Int {
  match rule {
    HoistPagedKeyValuePageLookup => 1
    FuseScoreTransform => 2
    FuseOnlineSoftmaxFold => 3
    ReuseOnlineSoftmaxTileStorage => 4
    PartitionOnlineSoftmaxFold => 5
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
  parallel_workgroup_target? : Int = 32,
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
