///|
fn projection_tile_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
pub fn select_projection_tile(
  request : ProjectionTileCompilerRequest,
) -> SelectedProjectionTileCompilation {
  { request, }
}

///|
fn projection_expression_code(expression : ProjectionTileExpression) -> Int {
  match expression {
    ProjectionTileInput => 1
    ProjectionTileWeight(ProjectionDenseWeight) => 2
    ProjectionSelectQueryRowEnds(_) => 3
    ProjectionDot(_, _) => 4
    ProjectionStore(_) => 5
    ProjectionTileWeight(ProjectionGateWeight) => 6
    ProjectionTileWeight(ProjectionUpWeight) => 7
    ProjectionTileWeight(ProjectionDownWeight) => 8
    ProjectionSiluGate(_, _) => 9
    ProjectionTileWeight(ProjectionQueryNormWeight) => 10
    ProjectionTileWeight(ProjectionKeyNormWeight) => 11
    ProjectionQkHeadRmsNorm(_, _, _) => 12
    ProjectionPositionedRotary(_) => 13
    ProjectionStoreAndCommitPagedKv(_) => 14
  }
}

///|
fn projection_expression_operands(
  expression : ProjectionTileExpression,
) -> Array[ProjectionTileValueId] {
  match expression {
    ProjectionTileInput | ProjectionTileWeight(_) => []
    ProjectionSelectQueryRowEnds(input) | ProjectionStore(input) => [input]
    ProjectionDot(left, right) | ProjectionSiluGate(left, right) =>
      [left, right]
    ProjectionQkHeadRmsNorm(value, query_weight, key_weight) =>
      [value, query_weight, key_weight]
    ProjectionPositionedRotary(value)
    | ProjectionStoreAndCommitPagedKv(value) => [value]
  }
}

///|
fn projection_program_canonical(
  request : ProjectionTileCompilerRequest,
  bindings : ArrayView[ProjectionTileBinding],
  root : ProjectionTileValueId,
) -> Bytes {
  let (input_width, output_width, intermediate_width) = request.dispatch
    .shape()
    .dimensions()
  let builder = StringBuilder()
  builder <+ "luna-projection-tile-program-v2\n"
  builder <+ "shape=\{input_width},\{output_width},\{intermediate_width}\n"
  builder <+ "demand=\{if request.demand == QueryRowEnds { 2 } else { 1 }}\n"
  builder <+ "bounds=\{request.maximum_rows},\{request.maximum_tokens}\n"
  if request.single_row_numerics == AllowDeterministicDotTree {
    builder <+ "single_row_numerics=deterministic-dot-tree\n"
  }
  match request.epilogue {
    ProjectionStoreRows => builder <+ "epilogue=store\n"
    ProjectionAttentionIngress(spec) =>
      builder <+
        "epilogue=attention-ingress,\{spec.query_heads},\{spec.kv_heads},\{spec.head_dimension},\{spec.norm_epsilon},\{spec.rope_theta}\n"
  }
  for binding in bindings {
    builder <+
      "binding=\{binding.id.0},\{projection_expression_code(binding.expression)}"
    for operand in projection_expression_operands(binding.expression) {
      builder <+ ",\{operand.0}"
    }
    builder <+ "\n"
  }
  builder <+ "root=\{root.0}\n"
  @utf8.encode(builder.to_string())
}

///|
pub fn elaborate_projection_tile(
  selected : SelectedProjectionTileCompilation,
) -> SemanticProjectionTileCompilation {
  let input = ProjectionTileBinding::{
    id: projection_tile_value_id(0),
    kind: ProjectionInputRows,
    expression: ProjectionTileInput,
  }
  let (bindings, root) = if selected.request.dispatch.shape().family() ==
    GatedMlp {
    let gate_weight = ProjectionTileBinding::{
      id: projection_tile_value_id(1),
      kind: ProjectionWeights,
      expression: ProjectionTileWeight(ProjectionGateWeight),
    }
    let up_weight = ProjectionTileBinding::{
      id: projection_tile_value_id(2),
      kind: ProjectionWeights,
      expression: ProjectionTileWeight(ProjectionUpWeight),
    }
    let down_weight = ProjectionTileBinding::{
      id: projection_tile_value_id(3),
      kind: ProjectionWeights,
      expression: ProjectionTileWeight(ProjectionDownWeight),
    }
    let gate_dot = ProjectionTileBinding::{
      id: projection_tile_value_id(4),
      kind: ProjectionAccumulator,
      expression: ProjectionDot(input.id, gate_weight.id),
    }
    let up_dot = ProjectionTileBinding::{
      id: projection_tile_value_id(5),
      kind: ProjectionAccumulator,
      expression: ProjectionDot(input.id, up_weight.id),
    }
    let activated = ProjectionTileBinding::{
      id: projection_tile_value_id(6),
      kind: ProjectionIntermediateRows,
      expression: ProjectionSiluGate(gate_dot.id, up_dot.id),
    }
    let down_dot = ProjectionTileBinding::{
      id: projection_tile_value_id(7),
      kind: ProjectionAccumulator,
      expression: ProjectionDot(activated.id, down_weight.id),
    }
    let store = ProjectionTileBinding::{
      id: projection_tile_value_id(8),
      kind: ProjectionOutputRows,
      expression: ProjectionStore(down_dot.id),
    }
    (
      [
        input, gate_weight, up_weight, down_weight, gate_dot, up_dot, activated,
        down_dot, store,
      ],
      store.id,
    )
  } else if selected.request.epilogue is ProjectionAttentionIngress(_) {
    guard selected.request.epilogue is ProjectionAttentionIngress(_) else {
      abort("checked attention ingress epilogue disappeared")
    }
    let weight = ProjectionTileBinding::{
      id: projection_tile_value_id(1),
      kind: ProjectionWeights,
      expression: ProjectionTileWeight(ProjectionDenseWeight),
    }
    let query_norm = ProjectionTileBinding::{
      id: projection_tile_value_id(2),
      kind: ProjectionWeights,
      expression: ProjectionTileWeight(ProjectionQueryNormWeight),
    }
    let key_norm = ProjectionTileBinding::{
      id: projection_tile_value_id(3),
      kind: ProjectionWeights,
      expression: ProjectionTileWeight(ProjectionKeyNormWeight),
    }
    let dot = ProjectionTileBinding::{
      id: projection_tile_value_id(4),
      kind: ProjectionAccumulator,
      expression: ProjectionDot(input.id, weight.id),
    }
    let normalized = ProjectionTileBinding::{
      id: projection_tile_value_id(5),
      kind: ProjectionNormalizedRows,
      expression: ProjectionQkHeadRmsNorm(dot.id, query_norm.id, key_norm.id),
    }
    let positioned = ProjectionTileBinding::{
      id: projection_tile_value_id(6),
      kind: ProjectionPositionedRows,
      expression: ProjectionPositionedRotary(normalized.id),
    }
    let committed = ProjectionTileBinding::{
      id: projection_tile_value_id(7),
      kind: ProjectionOutputRows,
      expression: ProjectionStoreAndCommitPagedKv(positioned.id),
    }
    (
      [
        input, weight, query_norm, key_norm, dot, normalized, positioned, committed,
      ],
      committed.id,
    )
  } else {
    let weight = ProjectionTileBinding::{
      id: projection_tile_value_id(1),
      kind: ProjectionWeights,
      expression: ProjectionTileWeight(ProjectionDenseWeight),
    }
    let bindings : Array[ProjectionTileBinding] = [input, weight]
    let dot_input = if selected.request.demand == QueryRowEnds {
      let selected_rows = ProjectionTileBinding::{
        id: projection_tile_value_id(2),
        kind: ProjectionSelectedRows,
        expression: ProjectionSelectQueryRowEnds(input.id),
      }
      bindings.push(selected_rows)
      selected_rows.id
    } else {
      input.id
    }
    let dot = ProjectionTileBinding::{
      id: projection_tile_value_id(bindings.length()),
      kind: ProjectionAccumulator,
      expression: ProjectionDot(dot_input, weight.id),
    }
    bindings.push(dot)
    let store = ProjectionTileBinding::{
      id: projection_tile_value_id(bindings.length()),
      kind: ProjectionOutputRows,
      expression: ProjectionStore(dot.id),
    }
    bindings.push(store)
    (bindings, store.id)
  }
  let canonical = projection_program_canonical(selected.request, bindings, root)
  let program = ProjectionTileProgram::{
    shape: selected.request.dispatch.shape(),
    demand: selected.request.demand,
    epilogue: selected.request.epilogue,
    maximum_rows: selected.request.maximum_rows,
    maximum_tokens: selected.request.maximum_tokens,
    bindings: ReadOnlyArray::from_array(bindings),
    root,
    canonical,
    digest: ProjectionTileProgramDigest(projection_tile_sha256(canonical)),
  }
  { selected, program }
}

///|
fn projection_rewrite_code(rule : ProjectionTileRewriteRule) -> Int {
  match rule {
    PruneUnobservedTokenRows => 1
    HoistSelectedInputTile => 2
    ReuseInputTileAcrossOutputTiles => 3
    FuseSiblingInputTraversals => 4
    ReuseIntermediateTileAcrossOutputTiles => 5
    FuseAttentionIngressEpilogue => 6
    ElideProjectedQkvRoundTrip => 7
    HoistRotaryInverseFrequencies => 8
    FuseRotaryPairEvaluation => 9
  }
}

///|
pub fn optimize_projection_tile(
  semantic : SemanticProjectionTileCompilation,
) -> OptimizedProjectionTileCompilation {
  let normalized = normalize_projection_tile(semantic)
  let program = normalized.program
  let selected = program.demand == QueryRowEnds
  let facts = projection_dataflow_facts(program)
  let gated_mlp = facts.gated_intermediate
  // Input reuse is a semantic sharing opportunity only when the
  // functional graph has multiple consumers of the same input value. A lone
  // projection has one Dot consumer; staging its input across output tiles is
  // a backend scheduling candidate whose synchronization cost must be proven,
  // not an unconditional functional rewrite.
  let shared_input_value = facts.shared_dot_input
  let trace = [
    { rule: PruneUnobservedTokenRows, changed: selected },
    { rule: HoistSelectedInputTile, changed: selected },
    { rule: ReuseInputTileAcrossOutputTiles, changed: shared_input_value },
    { rule: FuseSiblingInputTraversals, changed: gated_mlp },
    { rule: ReuseIntermediateTileAcrossOutputTiles, changed: gated_mlp },
  ]
  let fused_attention_ingress = facts.exclusive_attention_ingress &&
    program.epilogue is ProjectionAttentionIngress(_)
  if fused_attention_ingress {
    trace.push({ rule: FuseAttentionIngressEpilogue, changed: true })
    trace.push({ rule: ElideProjectedQkvRoundTrip, changed: true })
    trace.push({ rule: HoistRotaryInverseFrequencies, changed: true })
    trace.push({ rule: FuseRotaryPairEvaluation, changed: true })
  }
  let trace = ReadOnlyArray::from_array(trace)
  let builder = StringBuilder()
  builder <+ "luna-projection-tile-optimization-v3\n"
  builder <+ "program_sha256=\{program.digest.0}\n"
  for step in trace {
    builder <+ "pass=\{projection_rewrite_code(step.rule)},\{step.changed}\n"
  }
  let canonical = @utf8.encode(builder.to_string())
  let optimization = ProjectionTileOptimization::{
    program_digest: program.digest,
    trace,
    canonical,
    digest: ProjectionTileOptimizationDigest(projection_tile_sha256(canonical)),
  }
  { semantic, normalized, optimization }
}

///|
fn projection_schedule_canonical(
  optimized : OptimizedProjectionTileCompilation,
  loops : ArrayView[ProjectionLoopCombinator],
  reuse_input_tile : Bool,
  single_row_reduction : ProjectionSingleRowReduction,
) -> Bytes {
  let builder = StringBuilder()
  let explicit_tree = optimized.semantic.selected.request.single_row_numerics ==
    AllowDeterministicDotTree
  if explicit_tree {
    builder <+ "luna-projection-tile-schedule-v5\n"
  } else {
    builder <+ "luna-projection-tile-schedule-v4\n"
  }
  builder <+ "program_sha256=\{optimized.normalized.program.digest.0}\n"
  builder <+ "optimization_sha256=\{optimized.optimization.digest.0}\n"
  for schedule_loop in loops {
    let kind = if schedule_loop.kind == ProjectionParallelMap { 1 } else { 2 }
    let axis = match schedule_loop.axis {
      ProjectionRowAxis => 1
      ProjectionOutputColumnAxis => 2
      ProjectionReductionAxis => 3
      ProjectionIntermediateColumnAxis => 4
      ProjectionIntermediateReductionAxis => 5
      ProjectionHeadAxis => 6
      ProjectionHeadComponentAxis => 7
    }
    builder <+
      "loop=\{kind},\{axis},\{schedule_loop.extent},\{schedule_loop.tile}\n"
  }
  builder <+ "reuse_input_tile=\{reuse_input_tile}\n"
  if explicit_tree {
    match single_row_reduction {
      OrderedDot => builder <+ "single_row_reduction=ordered\n"
      StridedPairwiseDot(lanes) =>
        builder <+ "single_row_reduction=strided-pairwise,\{lanes}\n"
    }
  }
  builder <+
    "reuse_intermediate_tile=\{optimized.optimization.applied(ReuseIntermediateTileAcrossOutputTiles)}\n"
  let rotary_optimized = optimized.optimization.applied(
    FuseAttentionIngressEpilogue,
  )
  builder <+ "hoist_rotary_inverse_frequencies=\{rotary_optimized}\n"
  builder <+ "fuse_rotary_pairs=\{rotary_optimized}\n"
  @utf8.encode(builder.to_string())
}

///|
pub fn schedule_projection_tile(
  optimized : OptimizedProjectionTileCompilation,
) -> ScheduledProjectionTileCompilation {
  let request = optimized.semantic.selected.request
  let (input_width, output_width, _) = request.dispatch.shape().dimensions()
  let (tile_rows, tile_columns, tile_reduction) = request.dispatch
    .prefill()
    .tile()
    .dimensions()
  let row_extent = if request.demand == QueryRowEnds {
    request.maximum_rows
  } else {
    request.maximum_tokens
  }
  let row_loop = ProjectionLoopCombinator::{
    kind: ProjectionParallelMap,
    axis: ProjectionRowAxis,
    extent: row_extent,
    tile: tile_rows,
  }
  let loops : Array[ProjectionLoopCombinator] = if request.dispatch
    .shape()
    .family() ==
    GatedMlp {
    let (_, output_width, intermediate_width) = request.dispatch
      .shape()
      .dimensions()
    [
      row_loop,
      {
        kind: ProjectionParallelMap,
        axis: ProjectionIntermediateColumnAxis,
        extent: intermediate_width,
        tile: tile_columns,
      },
      {
        kind: ProjectionSequentialFold,
        axis: ProjectionReductionAxis,
        extent: input_width,
        tile: tile_reduction,
      },
      {
        kind: ProjectionParallelMap,
        axis: ProjectionOutputColumnAxis,
        extent: output_width,
        tile: tile_columns,
      },
      {
        kind: ProjectionSequentialFold,
        axis: ProjectionIntermediateReductionAxis,
        extent: intermediate_width,
        tile: tile_reduction,
      },
    ]
  } else {
    [
      row_loop,
      {
        kind: ProjectionParallelMap,
        axis: ProjectionOutputColumnAxis,
        extent: output_width,
        tile: tile_columns,
      },
      {
        kind: ProjectionSequentialFold,
        axis: ProjectionReductionAxis,
        extent: input_width,
        tile: tile_reduction,
      },
    ]
  }
  match request.epilogue {
    ProjectionAttentionIngress(spec) => {
      loops.push({
        kind: ProjectionParallelMap,
        axis: ProjectionHeadAxis,
        extent: spec.query_heads + 2 * spec.kv_heads,
        tile: 1,
      })
      loops.push({
        kind: ProjectionParallelMap,
        axis: ProjectionHeadComponentAxis,
        extent: spec.head_dimension,
        tile: spec.head_dimension,
      })
    }
    ProjectionStoreRows => ()
  }
  // Sibling gate/up dots share the same SSA input fragment already. Turning
  // that semantic reuse into a cross-warp shared-memory tile adds two block
  // barriers per reduction tile, so the conservative schedule keeps the
  // value in registers until a backend autotune record proves materializing
  // it is cheaper. The optimization trace still records the reuse opportunity.
  let reuse_input_tile = false
  let single_row_reduction = projection_single_row_reduction(request)
  let canonical = projection_schedule_canonical(
    optimized, loops, reuse_input_tile, single_row_reduction,
  )
  let schedule = ProjectionTileSchedule::{
    program_digest: optimized.normalized.program.digest,
    optimization_digest: optimized.optimization.digest,
    demand: request.demand,
    loops: ReadOnlyArray::from_array(loops),
    reuse_input_tile,
    single_row_reduction,
    reuse_intermediate_tile: optimized.optimization.applied(
      ReuseIntermediateTileAcrossOutputTiles,
    ),
    fuse_attention_ingress_epilogue: optimized.optimization.applied(
      FuseAttentionIngressEpilogue,
    ),
    hoist_rotary_inverse_frequencies: optimized.optimization.applied(
      HoistRotaryInverseFrequencies,
    ),
    fuse_rotary_pairs: optimized.optimization.applied(FuseRotaryPairEvaluation),
    canonical,
    digest: ProjectionTileScheduleDigest(projection_tile_sha256(canonical)),
  }
  let compilation_canonical = @utf8.encode(
    "luna-projection-functional-compilation-v4\n" +
    "program_sha256=" +
    optimized.normalized.program.digest.0 +
    "\noptimization_sha256=" +
    optimized.optimization.digest.0 +
    "\nschedule_sha256=" +
    schedule.digest.0 +
    "\n",
  )
  {
    optimized,
    schedule,
    canonical: compilation_canonical,
    digest: ProjectionTileCompilationDigest(
      projection_tile_sha256(compilation_canonical),
    ),
  }
}

///|
pub fn compile_projection_tile(
  request : ProjectionTileCompilerRequest,
) -> ScheduledProjectionTileCompilation {
  schedule_projection_tile(
    optimize_projection_tile(
      elaborate_projection_tile(select_projection_tile(request)),
    ),
  )
}
