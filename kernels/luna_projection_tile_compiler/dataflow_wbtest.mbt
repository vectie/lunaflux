///|
fn dataflow_test_program(
  bindings : Array[ProjectionTileBinding],
  root : Int,
) -> ProjectionTileProgram raise {
  let request = ProjectionTileCompilerRequest::new(
    dispatch=projection_compiler_dispatch(Qkv),
    maximum_rows=32,
    maximum_tokens=256,
    demand=AllTokenRows,
  )
  let base = elaborate_projection_tile(select_projection_tile(request)).program
  let root = projection_tile_value_id(root)
  let canonical = projection_program_canonical(request, bindings, root)
  {
    ..base,
    bindings: ReadOnlyArray::from_array(bindings),
    root,
    canonical,
    digest: ProjectionTileProgramDigest(projection_tile_sha256(canonical)),
  }
}

///|
fn dataflow_test_binding(
  id : Int,
  kind : ProjectionTileValueKind,
  expression : ProjectionTileExpression,
) -> ProjectionTileBinding {
  { id: projection_tile_value_id(id), kind, expression }
}

///|
test "dot reuse follows live consumers rather than the family tag" {
  let id = projection_tile_value_id
  let bindings = [
    dataflow_test_binding(0, ProjectionInputRows, ProjectionTileInput),
    dataflow_test_binding(
      1,
      ProjectionWeights,
      ProjectionTileWeight(ProjectionDenseWeight),
    ),
    dataflow_test_binding(2, ProjectionAccumulator, ProjectionDot(id(0), id(1))),
    dataflow_test_binding(3, ProjectionAccumulator, ProjectionDot(id(0), id(1))),
    dataflow_test_binding(4, ProjectionOutputRows, ProjectionStore(id(2))),
  ]
  let dead_sibling = dataflow_test_program(bindings, 4)
  assert_false(projection_dataflow_facts(dead_sibling).shared_dot_input)
  // A second store is an observable root, even though program.root remains 4.
  bindings.push(
    dataflow_test_binding(5, ProjectionOutputRows, ProjectionStore(id(3))),
  )
  let live_sibling = dataflow_test_program(bindings, 4)
  assert_true(projection_dataflow_facts(live_sibling).shared_dot_input)
  assert_true(projection_live_values(live_sibling).contains(id(3)))
  assert_eq(
    projection_use_count(
      live_sibling,
      projection_live_values(live_sibling),
      id(0),
    ),
    2,
  )
}

///|
test "ingress round trip elimination requires exclusive intermediate use" {
  let id = projection_tile_value_id
  let bindings = [
    dataflow_test_binding(0, ProjectionInputRows, ProjectionTileInput),
    dataflow_test_binding(
      1,
      ProjectionWeights,
      ProjectionTileWeight(ProjectionDenseWeight),
    ),
    dataflow_test_binding(2, ProjectionAccumulator, ProjectionDot(id(0), id(1))),
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
  ]
  assert_true(
    projection_dataflow_facts(dataflow_test_program(bindings, 5)).exclusive_attention_ingress,
  )
  bindings.push(
    dataflow_test_binding(6, ProjectionOutputRows, ProjectionStore(id(2))),
  )
  assert_false(
    projection_dataflow_facts(dataflow_test_program(bindings, 5)).exclusive_attention_ingress,
  )
}

///|
test "demand propagation is independent of dense value numbering" {
  let id = projection_tile_value_id
  let program = dataflow_test_program(
    [
      dataflow_test_binding(11, ProjectionInputRows, ProjectionTileInput),
      dataflow_test_binding(
        27,
        ProjectionWeights,
        ProjectionTileWeight(ProjectionDenseWeight),
      ),
      dataflow_test_binding(
        39,
        ProjectionAccumulator,
        ProjectionDot(id(11), id(27)),
      ),
      dataflow_test_binding(88, ProjectionOutputRows, ProjectionStore(id(39))),
    ],
    88,
  )
  assert_eq(projection_live_values(program).length(), 4)
  assert_false(projection_dataflow_facts(program).shared_dot_input)
}

///|
test "schedule consumes exclusive-use analysis rather than the epilogue tag" {
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
  let original = elaborate_projection_tile(select_projection_tile(request))
  let bindings = [ for binding in original.program.bindings => binding ]
  bindings.push(
    dataflow_test_binding(
      8,
      ProjectionOutputRows,
      ProjectionStore(projection_tile_value_id(4)),
    ),
  )
  let canonical = projection_program_canonical(
    request,
    bindings,
    original.program.root,
  )
  let semantic = {
    ..original,
    program: {
      ..original.program,
      bindings: ReadOnlyArray::from_array(bindings),
      canonical,
      digest: ProjectionTileProgramDigest(projection_tile_sha256(canonical)),
    },
  }
  let optimized = optimize_projection_tile(semantic)
  let scheduled = schedule_projection_tile(optimized)
  assert_false(optimized.optimization.applied(ElideProjectedQkvRoundTrip))
  assert_false(scheduled.schedule.fuses_attention_ingress_epilogue())
  assert_false(scheduled.schedule.hoists_rotary_inverse_frequencies())
  assert_true(
    projection_live_values(semantic.program).contains(
      projection_tile_value_id(8),
    ),
  )
}
