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
fn apply_rewrite_pass(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
  rule : AttentionTileRewriteRule,
) -> AttentionRewritePassResult {
  match rule {
    FuseScoreTransform => apply_score_transform_pass(program, regions)
    FuseOnlineSoftmaxFold => apply_online_softmax_fold_pass(program, regions)
  }
}

///|
fn run_rewrite_pipeline(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  passes : ReadOnlyArray[AttentionTileRewriteRule],
  index : Int,
  state : AttentionRewritePipelineState,
) -> AttentionRewritePipelineState {
  if index == passes.length() {
    return state
  }
  let pass = passes[index]
  let result = apply_rewrite_pass(program, state.regions, pass)
  run_rewrite_pipeline(program, passes, index + 1, {
    regions: result.regions,
    trace: append_trace(state.trace, {
      rule: pass,
      changed: result.changed,
      region_count: result.regions.length(),
    }),
  })
}

///|
fn rewrite_rule_code(rule : AttentionTileRewriteRule) -> Int {
  match rule {
    FuseScoreTransform => 1
    FuseOnlineSoftmaxFold => 2
  }
}

///|
fn canonical_optimization(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  regions : ReadOnlyArray[AttentionTileRewriteRegion],
  trace : ReadOnlyArray[AttentionTileRewriteStep],
) -> Bytes {
  let builder = StringBuilder()
  builder <+ "luna-attention-tile-optimization-v1\n"
  builder <+ "program_sha256=\{program.digest().as_hex()}\n"
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
) -> AttentionTileOptimization raise AttentionTileOptimizerError {
  let passes = ReadOnlyArray::from_array([
    FuseScoreTransform,
    FuseOnlineSoftmaxFold,
  ])
  let state = run_rewrite_pipeline(program, passes, 0, {
    regions: ReadOnlyArray::from_array([]),
    trace: ReadOnlyArray::from_array([]),
  })
  if state.trace.length() != passes.length() || state.regions.length() == 0 {
    raise InvalidAttentionTileOptimization(InvalidRewritePipeline)
  }
  let canonical = canonical_optimization(program, state.regions, state.trace)
  {
    program_digest: program.digest(),
    regions: state.regions,
    trace: state.trace,
    canonical,
    digest: AttentionTileOptimizationDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
