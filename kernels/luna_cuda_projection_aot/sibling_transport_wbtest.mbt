///|
test "coowned producer alternatives lower the same arithmetic and launch" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let dispatch = runtime.dispatch_plan(1024)
  let choice = @luna_projection_tile_compiler.ProjectionFoldChoice::new(
    SiblingFold,
    stages=2,
    transfer_tiles=2,
    fragment_stages=1,
    row_tiles=4,
    output_tiles=8,
    consumer_groups=8,
  )
  let compilation = projection_row_variant_compilation(
    parameters,
    dispatch,
    32,
    1024,
    fold_choices=[choice],
  )
  let sources = [
    @luna_projection_tile_compiler.IndependentSiblingTransport,
    SegmentedSiblingTransport,
  ].map(transport => {
    let physical = projection_refine_physical(
      compilation,
      sibling_ownership=CoownedSiblingValues,
      sibling_transport=transport,
    )
    let sibling = physical.sibling.unwrap()
    assert_eq(sibling.distribution.consumer_groups(), 8)
    assert_eq(sibling.schedule.row_width(), 64)
    assert_eq(sibling.schedule.output_tiles(), 8)
    assert_eq(physical.sibling_epilogue.unwrap().result_planes(), 0)
    let source = projection_sibling_fold_pipeline_source(
      sibling,
      parameters,
      physical.materialization,
      epilogue=physical.sibling_epilogue.unwrap(),
    )
    assert_true(source.contains("float cg[4][8] = {}, cu[4][8] = {};"))
    assert_true(source.contains("(gate / (1.0f + expf(-gate))) * up"))
    assert_false(source.contains("gate_tiles"))
    assert_false(source.contains("peer_tiles"))
    assert_false(source.contains("cp.async"))
    if transport == SegmentedSiblingTransport {
      assert_true(source.contains("transfer_sources[5]"))
      assert_false(source.contains("bool overlap"))
      assert_true(source.contains("stage_operands(inner + 32, stage ^ 1);"))
    } else {
      assert_true(source.contains("bool overlap"))
      assert_false(source.contains("transfer_sources[5]"))
      assert_true(
        source.contains("stage_operands(inner + 32, stage ^ 1, true);"),
      )
    }
    source
  })
  assert_false(sources[0] == sources[1])
  // Explicit old transport remains byte-identical to the ordinary coowned
  // artifact. This is essential when comparing against preserved captures.
  let baseline = projection_refine_physical(
    compilation,
    sibling_ownership=CoownedSiblingValues,
  )
  assert_eq(
    sources[0],
    projection_sibling_fold_pipeline_source(
      baseline.sibling.unwrap(),
      parameters,
      baseline.materialization,
      epilogue=baseline.sibling_epilogue.unwrap(),
    ),
  )
}
