///|
/// Facts derived from value dependencies, not model or operation-family tags.
priv struct ProjectionDataflowFacts {
  shared_dot_input : Bool
  gated_intermediate : Bool
  exclusive_attention_ingress : Bool
}

///|
fn projection_binding(
  program : ProjectionTileProgram,
  id : ProjectionTileValueId,
) -> ProjectionTileBinding? {
  for binding in program.bindings {
    if binding.id == id {
      return Some(binding)
    }
  }
  None
}

///|
/// Stores are observable roots even when their returned value is unused.
/// A KV commit is never treated as a disposable pure expression.
fn projection_has_effect(expression : ProjectionTileExpression) -> Bool {
  match expression {
    ProjectionStore(_) | ProjectionStoreAndCommitPagedKv(_) => true
    ProjectionTileInput
    | ProjectionTileWeight(_)
    | ProjectionSelectQueryRowEnds(_)
    | ProjectionDot(_, _)
    | ProjectionSiluGate(_, _)
    | ProjectionQkHeadRmsNorm(_, _, _)
    | ProjectionPositionedRotary(_) => false
  }
}

///|
/// Topological bindings let one reverse fold propagate demand without a
/// recursive graph walk, global state, or an arbitrary iteration budget.
fn projection_live_values(
  program : ProjectionTileProgram,
) -> ReadOnlyArray[ProjectionTileValueId] {
  for i = program.bindings.length() - 1, live = ReadOnlyArray::from_array([
          program.root,
        ]); i >= 0; {
    let binding = program.bindings[i]
    let next = if projection_has_effect(binding.expression) ||
      live.contains(binding.id) {
      let extended = [ for id in live => id ]
      if !extended.contains(binding.id) {
        extended.push(binding.id)
      }
      for operand in projection_expression_operands(binding.expression) {
        if !extended.contains(operand) {
          extended.push(operand)
        }
      }
      ReadOnlyArray::from_array(extended)
    } else {
      live
    }
    continue i - 1, next
  } nobreak {
    live
  }
}

///|
fn projection_use_count(
  program : ProjectionTileProgram,
  live : ReadOnlyArray[ProjectionTileValueId],
  id : ProjectionTileValueId,
) -> Int {
  program.bindings.fold(init=if program.root == id { 1 } else { 0 }, (
    count,
    binding,
  ) => {
    if !live.contains(binding.id) {
      return count
    }
    projection_expression_operands(binding.expression).fold(init=count, (
      count,
      operand,
    ) => count + (if operand == id { 1 } else { 0 }))
  })
}

///|
fn projection_shared_dot_input(
  program : ProjectionTileProgram,
  live : ReadOnlyArray[ProjectionTileValueId],
) -> Bool {
  program.bindings.any(binding => {
    if !live.contains(binding.id) {
      return false
    }
    guard binding.expression is ProjectionDot(input, _) else { return false }
    program.bindings.any(other => {
      other.id != binding.id &&
      live.contains(other.id) &&
      other.expression is ProjectionDot(other_input, _) &&
      other_input == input
    })
  })
}

///|
fn projection_gated_intermediate(
  program : ProjectionTileProgram,
  live : ReadOnlyArray[ProjectionTileValueId],
) -> Bool {
  program.bindings.any(binding => {
    if !live.contains(binding.id) {
      return false
    }
    guard binding.expression is ProjectionDot(intermediate, _) &&
      projection_binding(program, intermediate) is Some(activated) &&
      activated.expression is ProjectionSiluGate(gate, up) &&
      projection_binding(program, gate) is Some(gate_dot) &&
      projection_binding(program, up) is Some(up_dot) &&
      gate_dot.expression is ProjectionDot(gate_input, _) &&
      up_dot.expression is ProjectionDot(up_input, _) else {
      return false
    }
    gate_input == up_input
  })
}

///|
/// Eliding projected buffers is legal only if the fused chain owns every
/// intermediate use. Additional live consumers retain the round trips.
fn projection_exclusive_attention_ingress(
  program : ProjectionTileProgram,
  live : ReadOnlyArray[ProjectionTileValueId],
) -> Bool {
  guard projection_binding(program, program.root) is Some(commit) &&
    commit.expression is ProjectionStoreAndCommitPagedKv(rotary_id) &&
    projection_binding(program, rotary_id) is Some(rotary) &&
    rotary.expression is ProjectionPositionedRotary(norm_id) &&
    projection_binding(program, norm_id) is Some(norm) &&
    norm.expression is ProjectionQkHeadRmsNorm(dot_id, _, _) &&
    projection_binding(program, dot_id) is Some(dot) &&
    dot.expression is ProjectionDot(_, _) else {
    return false
  }
  [dot_id, norm_id, rotary_id].all(id => {
    projection_use_count(program, live, id) == 1
  })
}

///|
fn projection_dataflow_facts(
  program : ProjectionTileProgram,
) -> ProjectionDataflowFacts {
  let live = projection_live_values(program)
  {
    shared_dot_input: projection_shared_dot_input(program, live),
    gated_intermediate: projection_gated_intermediate(program, live),
    exclusive_attention_ingress: projection_exclusive_attention_ingress(
      program, live,
    ),
  }
}

///|
fn ProjectionTileOptimization::applied(
  self : ProjectionTileOptimization,
  rule : ProjectionTileRewriteRule,
) -> Bool {
  self.trace.any(step => step.rule == rule && step.changed)
}
