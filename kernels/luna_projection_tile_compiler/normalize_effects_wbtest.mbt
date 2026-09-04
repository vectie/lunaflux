///|
test "input reads and dot values are not reused across a store barrier" {
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
      dataflow_test_binding(3, ProjectionOutputRows, ProjectionStore(id(2))),
      dataflow_test_binding(4, ProjectionInputRows, ProjectionTileInput),
      dataflow_test_binding(
        5,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionDenseWeight),
      ),
      dataflow_test_binding(
        6,
        ProjectionAccumulator,
        ProjectionDot(id(4), id(5)),
      ),
      dataflow_test_binding(7, ProjectionOutputRows, ProjectionStore(id(6))),
    ],
    7,
  )
  let normalized = normalize_projection_tile(
    normalization_test_semantic(program),
  )
  assert_eq(normalized.program, program)
  assert_eq(normalized.common_subexpressions, 0)
}

///|
test "CSE does not conflate bindings with distinct value kinds" {
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
        ProjectionIntermediateRows,
        ProjectionDot(id(0), id(1)),
      ),
      dataflow_test_binding(4, ProjectionOutputRows, ProjectionStore(id(2))),
      dataflow_test_binding(5, ProjectionOutputRows, ProjectionStore(id(3))),
    ],
    5,
  )
  let normalized = normalize_projection_tile(
    normalization_test_semantic(program),
  )
  assert_eq(normalized.program, program)
  assert_eq(normalized.common_subexpressions, 0)
}

///|
test "normalization rewrites the full norm rotary chain and preserves both effects" {
  let id = projection_tile_value_id
  let program = dataflow_test_program(
    [
      // Dead leading binding forces EVERY surviving operand to be renumbered.
      dataflow_test_binding(
        0,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionUpWeight),
      ),
      dataflow_test_binding(11, ProjectionInputRows, ProjectionTileInput),
      dataflow_test_binding(
        22,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionDenseWeight),
      ),
      dataflow_test_binding(
        33,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionQueryNormWeight),
      ),
      dataflow_test_binding(
        44,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionKeyNormWeight),
      ),
      dataflow_test_binding(
        55,
        ProjectionSelectedRows,
        ProjectionSelectQueryRowEnds(id(11)),
      ),
      dataflow_test_binding(
        66,
        ProjectionAccumulator,
        ProjectionDot(id(55), id(22)),
      ),
      dataflow_test_binding(
        77,
        ProjectionNormalizedRows,
        ProjectionQkHeadRmsNorm(id(66), id(33), id(44)),
      ),
      dataflow_test_binding(
        88,
        ProjectionNormalizedRows,
        ProjectionQkHeadRmsNorm(id(66), id(33), id(44)),
      ),
      dataflow_test_binding(
        99,
        ProjectionPositionedRows,
        ProjectionPositionedRotary(id(77)),
      ),
      dataflow_test_binding(
        111,
        ProjectionPositionedRows,
        ProjectionPositionedRotary(id(88)),
      ),
      dataflow_test_binding(122, ProjectionOutputRows, ProjectionStore(id(99))),
      dataflow_test_binding(
        133,
        ProjectionOutputRows,
        ProjectionStoreAndCommitPagedKv(id(111)),
      ),
    ],
    133,
  )
  let normalized = normalize_projection_tile(
    normalization_test_semantic(program),
  )
  assert_eq(normalized.common_subexpressions, 2)
  assert_eq(normalized.dead_bindings, 1)
  assert_eq(normalized.program.bindings.length(), 10)
  assert_eq(
    normalization_test_observations(program),
    normalization_test_observations(normalized.program),
  )
  assert_eq(
    normalized.program.bindings[6].expression,
    ProjectionQkHeadRmsNorm(id(5), id(2), id(3)),
  )
  assert_eq(
    normalized.program.bindings[7].expression,
    ProjectionPositionedRotary(id(6)),
  )
  assert_eq(normalized.program.bindings[8].expression, ProjectionStore(id(7)))
  assert_eq(
    normalized.program.bindings[9].expression,
    ProjectionStoreAndCommitPagedKv(id(7)),
  )
}
