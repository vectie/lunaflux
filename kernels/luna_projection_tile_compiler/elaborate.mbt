///|
fn projection_expression_code(expression : ProjectionTileExpression) -> Int32 {
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
  for segment in request.concatenated_weights {
    builder <+
      "weight-row-segment=\{segment.operand},\{segment.row_begin},\{segment.row_end}\n"
  }
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
    concatenated_weights: selected.request.concatenated_weights,
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
  { selected, program, }
}
