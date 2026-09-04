///|
/// Source-dependent diagnostics are not part of the normalized code identity.
/// The original semantic graph remains in OptimizedProjectionTileCompilation.
priv struct ProjectionTileNormalization {
  program : ProjectionTileProgram
  common_subexpressions : Int
  dead_bindings : Int
} derive(Debug, Eq)

///|
fn projection_remap_expression(
  expression : ProjectionTileExpression,
  rename : (ProjectionTileValueId) -> ProjectionTileValueId,
) -> ProjectionTileExpression {
  match expression {
    ProjectionTileInput | ProjectionTileWeight(_) => expression
    ProjectionSelectQueryRowEnds(value) =>
      ProjectionSelectQueryRowEnds(rename(value))
    ProjectionDot(left, right) => ProjectionDot(rename(left), rename(right))
    ProjectionSiluGate(gate, up) => ProjectionSiluGate(rename(gate), rename(up))
    ProjectionQkHeadRmsNorm(value, query, key) =>
      ProjectionQkHeadRmsNorm(rename(value), rename(query), rename(key))
    ProjectionPositionedRotary(value) =>
      ProjectionPositionedRotary(rename(value))
    ProjectionStoreAndCommitPagedKv(value) =>
      ProjectionStoreAndCommitPagedKv(rename(value))
    ProjectionStore(value) => ProjectionStore(rename(value))
  }
}

///|
/// Demand pruning followed by typed, exact value numbering. Bindings are
/// topological SSA from elaboration. Local construction tables do not escape;
/// neither the input graph nor any shared compiler state is mutated.
///
/// Operand order is significant: no reassociation, commutation, contraction,
/// or approximate numeric equality. Without an alias model, every observable
/// effect ends the CSE region, including for input/weight reads. Effects are
/// never merged, dropped, or reordered even when they consume the same value.
fn normalize_projection_tile(
  semantic : SemanticProjectionTileCompilation,
) -> ProjectionTileNormalization {
  let source = semantic.program
  let live = projection_live_values(source)
  let aliases : Map[ProjectionTileValueId, ProjectionTileValueId] = Map([])
  let mut available : Map[
    (ProjectionTileValueKind, ProjectionTileExpression),
    ProjectionTileValueId,
  ] = Map([])
  let bindings : Array[ProjectionTileBinding] = []
  let mut common_subexpressions = 0
  let mut dead_bindings = 0
  let rename = (id : ProjectionTileValueId) => {
    guard aliases.get(id) is Some(mapped) else {
      abort("projection normalization requires topological SSA")
    }
    mapped
  }
  for binding in source.bindings {
    if !live.contains(binding.id) {
      dead_bindings += 1
      continue
    }
    let expression = projection_remap_expression(binding.expression, rename)
    let effect = projection_has_effect(expression)
    let key = (binding.kind, expression)
    if !effect && available.get(key) is Some(existing) {
      aliases.set(binding.id, existing)
      common_subexpressions += 1
      continue
    }
    let id = projection_tile_value_id(bindings.length())
    bindings.push({ id, kind: binding.kind, expression })
    aliases.set(binding.id, id)
    if effect {
      available = Map([])
    } else {
      available.set(key, id)
    }
  }
  let root = rename(source.root)
  let canonical = projection_program_canonical(
    semantic.selected.request,
    bindings,
    root,
  )
  {
    program: {
      ..source,
      bindings: ReadOnlyArray::from_array(bindings),
      root,
      canonical,
      digest: ProjectionTileProgramDigest(projection_tile_sha256(canonical)),
    },
    common_subexpressions,
    dead_bindings,
  }
}
