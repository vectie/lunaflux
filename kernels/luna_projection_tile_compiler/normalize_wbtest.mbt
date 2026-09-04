///|
fn normalization_test_semantic(
  program : ProjectionTileProgram,
) -> SemanticProjectionTileCompilation raise {
  let request = ProjectionTileCompilerRequest::new(
    dispatch=projection_compiler_dispatch(Qkv),
    maximum_rows=32,
    maximum_tokens=256,
    demand=AllTokenRows,
  )
  { selected: select_projection_tile(request), program }
}

///|
/// Interpret the graph in a free symbolic algebra: pure operations preserve
/// their exact ordered operands; effects additionally append to an ordered
/// observation log. This checks substitution, not GPU numerical performance.
fn normalization_test_observations(
  program : ProjectionTileProgram,
) -> (String, Array[String]) {
  let values : Map[ProjectionTileValueId, String] = Map([])
  let effects : Array[String] = []
  for binding in program.bindings {
    let builder = StringBuilder()
    builder <+ "\{projection_expression_code(binding.expression)}("
    for operand in projection_expression_operands(binding.expression) {
      builder <+ "\{values[operand]},"
    }
    builder <+ ")"
    let value = builder.to_string()
    values.set(binding.id, value)
    if projection_has_effect(binding.expression) {
      effects.push(value)
    }
  }
  (values[program.root], effects)
}

///|
test "normalization merges cascading exact expressions and prunes dead work" {
  let id = projection_tile_value_id
  let program = dataflow_test_program(
    [
      dataflow_test_binding(0, ProjectionInputRows, ProjectionTileInput),
      dataflow_test_binding(
        1,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionDenseWeight),
      ),
      dataflow_test_binding(
        2,
        ProjectionAccumulator,
        ProjectionDot(id(0), id(1)),
      ),
      dataflow_test_binding(
        3,
        ProjectionAccumulator,
        ProjectionDot(id(0), id(1)),
      ),
      dataflow_test_binding(
        4,
        ProjectionIntermediateRows,
        ProjectionSiluGate(id(2), id(2)),
      ),
      dataflow_test_binding(
        5,
        ProjectionIntermediateRows,
        ProjectionSiluGate(id(3), id(3)),
      ),
      dataflow_test_binding(
        6,
        ProjectionPositionedRows,
        ProjectionPositionedRotary(id(2)),
      ),
      dataflow_test_binding(7, ProjectionOutputRows, ProjectionStore(id(4))),
      dataflow_test_binding(8, ProjectionOutputRows, ProjectionStore(id(5))),
    ],
    8,
  )
  let semantic = normalization_test_semantic(program)
  let compiled = schedule_projection_tile(optimize_projection_tile(semantic))
  assert_eq(compiled.eliminated_bindings(), (2, 1))
  assert_eq(compiled.program().bindings.length(), 6)
  assert_eq(compiled.source_program(), program)
  assert_eq(program.bindings.length(), 9)
  assert_eq(
    normalization_test_observations(program),
    normalization_test_observations(compiled.program()),
  )
  // Reuse analysis must run AFTER CSE; there is now only one live dot.
  assert_false(compiled.optimization().applied(ReuseInputTileAcrossOutputTiles))
  let again = schedule_projection_tile(
    optimize_projection_tile({ ..semantic, program: compiled.program() }),
  )
  assert_eq(again.program(), compiled.program())
  assert_eq(again.eliminated_bindings(), (0, 0))
  assert_eq(again.digest(), compiled.digest())
}

///|
test "alpha renaming and dead insertions converge without erasing source provenance" {
  let id = projection_tile_value_id
  let request = ProjectionTileCompilerRequest::new(
    dispatch=projection_compiler_dispatch(Qkv),
    maximum_rows=32,
    maximum_tokens=256,
    demand=AllTokenRows,
  )
  let source = elaborate_projection_tile(select_projection_tile(request))
  let changed = dataflow_test_program(
    [
      dataflow_test_binding(11, ProjectionInputRows, ProjectionTileInput),
      dataflow_test_binding(
        27,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionDenseWeight),
      ),
      dataflow_test_binding(
        33,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionUpWeight),
      ),
      dataflow_test_binding(
        39,
        ProjectionAccumulator,
        ProjectionDot(id(11), id(27)),
      ),
      dataflow_test_binding(
        51,
        ProjectionAccumulator,
        ProjectionDot(id(11), id(27)),
      ),
      dataflow_test_binding(88, ProjectionOutputRows, ProjectionStore(id(51))),
    ],
    88,
  )
  let original = schedule_projection_tile(optimize_projection_tile(source))
  let normalized = schedule_projection_tile(
    optimize_projection_tile({ ..source, program: changed }),
  )
  assert_eq(normalized.program(), original.program())
  assert_eq(normalized.digest(), original.digest())
  assert_eq(normalized.eliminated_bindings(), (0, 2))
  assert_not_eq(
    normalized.source_program().digest(),
    original.source_program().digest(),
  )
}

///|
test "effects remain ordered observable roots and block commoning across commits" {
  let id = projection_tile_value_id
  let program = dataflow_test_program(
    [
      dataflow_test_binding(0, ProjectionInputRows, ProjectionTileInput),
      dataflow_test_binding(
        1,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionDenseWeight),
      ),
      dataflow_test_binding(
        2,
        ProjectionAccumulator,
        ProjectionDot(id(0), id(1)),
      ),
      dataflow_test_binding(
        3,
        ProjectionNormalizedRows,
        ProjectionQkHeadRmsNorm(id(2), id(1), id(1)),
      ),
      dataflow_test_binding(
        4,
        ProjectionPositionedRows,
        ProjectionPositionedRotary(id(3)),
      ),
      dataflow_test_binding(
        5,
        ProjectionOutputRows,
        ProjectionStoreAndCommitPagedKv(id(4)),
      ),
      dataflow_test_binding(
        6,
        ProjectionOutputRows,
        ProjectionStoreAndCommitPagedKv(id(4)),
      ),
      dataflow_test_binding(
        7,
        ProjectionAccumulator,
        ProjectionDot(id(0), id(1)),
      ),
      dataflow_test_binding(8, ProjectionOutputRows, ProjectionStore(id(7))),
      dataflow_test_binding(9, ProjectionOutputRows, ProjectionStore(id(7))),
    ],
    0,
  )
  let normalized = normalize_projection_tile(
    normalization_test_semantic(program),
  )
  assert_eq(normalized.common_subexpressions, 0)
  assert_eq(normalized.dead_bindings, 0)
  assert_eq(normalized.program, program)
  assert_eq(normalization_test_observations(normalized.program).1.length(), 4)
}

///|
test "CSE preserves operand order and distinct weight roles" {
  let id = projection_tile_value_id
  let program = dataflow_test_program(
    [
      dataflow_test_binding(0, ProjectionInputRows, ProjectionTileInput),
      dataflow_test_binding(
        1,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionGateWeight),
      ),
      dataflow_test_binding(
        2,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionUpWeight),
      ),
      dataflow_test_binding(
        3,
        ProjectionAccumulator,
        ProjectionDot(id(0), id(1)),
      ),
      dataflow_test_binding(
        4,
        ProjectionAccumulator,
        ProjectionDot(id(0), id(2)),
      ),
      dataflow_test_binding(
        5,
        ProjectionIntermediateRows,
        ProjectionSiluGate(id(3), id(4)),
      ),
      dataflow_test_binding(
        6,
        ProjectionIntermediateRows,
        ProjectionSiluGate(id(4), id(3)),
      ),
      dataflow_test_binding(7, ProjectionOutputRows, ProjectionStore(id(5))),
      dataflow_test_binding(8, ProjectionOutputRows, ProjectionStore(id(6))),
    ],
    8,
  )
  let normalized = normalize_projection_tile(
    normalization_test_semantic(program),
  )
  assert_eq(normalized.program, program)
  assert_eq(normalized.common_subexpressions, 0)
}

///|
test "all elaborated families are normalization fixed points" {
  let families : Array[@luna_projection_strategy.ProjectionSemanticFamily] = [
    Dense,
    Qkv,
    LanguageModelHead,
    GatedMlp,
  ]
  for family in families {
    let request = ProjectionTileCompilerRequest::new(
      dispatch=projection_compiler_dispatch(family),
      maximum_rows=32,
      maximum_tokens=256,
      demand=if family == LanguageModelHead {
        QueryRowEnds
      } else {
        AllTokenRows
      },
    )
    let semantic = elaborate_projection_tile(select_projection_tile(request))
    let compiled = compile_projection_tile(request)
    assert_eq(compiled.program(), semantic.program)
    assert_eq(compiled.eliminated_bindings(), (0, 0))
  }
  let request = ProjectionTileCompilerRequest::with_attention_ingress_epilogue(
    dispatch=projection_compiler_dispatch(Qkv),
    maximum_rows=32,
    maximum_tokens=256,
    epilogue=ProjectionAttentionIngressEpilogue::new(
      query_heads=4,
      kv_heads=2,
      head_dimension=128,
      norm_epsilon=0.000001,
      rope_theta=1000000.0,
    ),
  )
  let semantic = elaborate_projection_tile(select_projection_tile(request))
  assert_eq(compile_projection_tile(request).program(), semantic.program)
}
