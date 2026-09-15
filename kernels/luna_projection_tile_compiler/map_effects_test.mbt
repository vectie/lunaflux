///|
test "map visibility and reader release preserve storage scopes" {
  let demands : Array[@luna_projection_tile_compiler.ProjectionResultDemand] = [
    AllTokenRows,
    QueryRowEnds,
  ]
  let input_boundaries : Array[
    @luna_projection_tile_compiler.ProjectionMapBoundary,
  ] = [InputWritten, InputRead]
  let result_boundaries : Array[
    @luna_projection_tile_compiler.ProjectionMapBoundary,
  ] = [ResultWritten, ResultRead]
  for demand in demands {
    let plan = demand.map_effect_plan()
    for boundary in input_boundaries {
      if demand == QueryRowEnds {
        assert_true(plan.at(boundary) is Some(WorkgroupFence))
      } else {
        assert_eq(plan.at(boundary), None)
      }
    }
    for boundary in result_boundaries {
      assert_true(plan.at(boundary) is Some(ConsumerGroupFence))
    }
  }
}
