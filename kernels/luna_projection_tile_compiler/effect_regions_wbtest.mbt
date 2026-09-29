///|
test "sibling traversal reuse cannot cross an observable store" {
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
      dataflow_test_binding(4, ProjectionOutputRows, ProjectionStore(id(0))),
      dataflow_test_binding(
        5,
        ProjectionAccumulator,
        ProjectionDot(id(0), id(2)),
      ),
      dataflow_test_binding(
        6,
        ProjectionIntermediateRows,
        ProjectionSiluGate(id(3), id(5)),
      ),
      dataflow_test_binding(
        7,
        ProjectionAccumulator,
        ProjectionDot(id(6), id(1)),
      ),
      dataflow_test_binding(8, ProjectionOutputRows, ProjectionStore(id(7))),
    ],
    8,
  )
  let facts = projection_dataflow_facts(program)
  assert_false(facts.shared_dot_input)
  assert_false(facts.gated_intermediate)
  let normalized = normalize_projection_tile(
    normalization_test_semantic(program),
  )
  assert_eq(
    normalization_test_observations(normalized.program),
    normalization_test_observations(program),
  )
  assert_false(projection_dataflow_facts(normalized.program).shared_dot_input)
}

///|
test "exclusive ingress use does not authorize moving across an unrelated effect" {
  let id = projection_tile_value_id
  let program = dataflow_test_program(
    [
      dataflow_test_binding(11, ProjectionInputRows, ProjectionTileInput),
      dataflow_test_binding(
        22,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionDenseWeight),
      ),
      dataflow_test_binding(
        33,
        ProjectionAccumulator,
        ProjectionDot(id(11), id(22)),
      ),
      dataflow_test_binding(44, ProjectionOutputRows, ProjectionStore(id(11))),
      dataflow_test_binding(
        55,
        ProjectionNormalizedRows,
        ProjectionQkHeadRmsNorm(id(33), id(22), id(22)),
      ),
      dataflow_test_binding(
        66,
        ProjectionPositionedRows,
        ProjectionPositionedRotary(id(55)),
      ),
      dataflow_test_binding(
        77,
        ProjectionOutputRows,
        ProjectionStoreAndCommitPagedKv(id(66)),
      ),
    ],
    77,
  )
  assert_false(projection_dataflow_facts(program).exclusive_attention_ingress)
  let regions = projection_effect_regions(program)
  assert_true(projection_same_effect_region(regions, [id(11), id(33), id(44)]))
  assert_true(projection_same_effect_region(regions, [id(55), id(66), id(77)]))
  assert_false(projection_same_effect_region(regions, [id(33), id(55)]))
  assert_false(projection_same_effect_region(regions, [id(100)]))
}
