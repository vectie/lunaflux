///|
test "explicit ring choices account for static plus dynamic storage before export" {
  let parameters = sibling_resource_test_parameters()
  let dispatch = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  ).dispatch_plan(1024)
  let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows=32,
    maximum_tokens=1024,
    demand=AllTokenRows,
  )
  for stages in [2, 3, 4] {
    let compilation = @luna_projection_tile_compiler.compile_projection_tile(
      request.with_fold_pipeline(
        SiblingFold,
        stages~,
        transfer_tiles=2,
        fragment_stages=1,
      ),
    )
    if stages == 2 {
      projection_validate_fold_resources(
        projection_refine_physical(compilation),
        16384,
      )
    } else {
      try
        projection_validate_fold_resources(
          projection_refine_physical(compilation),
          16384,
        )
      catch {
        LimitExceeded(LaunchDimensions) => ()
        _ => fail("unexpected resource rejection")
      } noraise {
        _ => fail("ring requiring unavailable opt-in was exported")
      }
    }
  }
}

///|
test "coowned 128 by 128 tile pays for its ring and not a nonexistent result plane" {
  let parameters = sibling_resource_test_parameters()
  let dispatch = projection_build_runtime_dispatch(
    parameters,
    2048,
    projection_fixture_target(),
    [],
  ).dispatch_plan(2048)
  let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows=32,
    maximum_tokens=2048,
    demand=AllTokenRows,
  )
  let scheduled = @luna_projection_tile_compiler.compile_projection_tile(
    request.with_fold_choices([
      @luna_projection_tile_compiler.ProjectionFoldChoice::new(
        SiblingFold,
        stages=2,
        transfer_tiles=2,
        fragment_stages=1,
        row_tiles=8,
        output_tiles=8,
        consumer_groups=8,
      ),
    ]),
  )
  let physical = projection_refine_physical(
    scheduled,
    sibling_ownership=CoownedSiblingValues,
  )
  assert_eq(physical.sibling_epilogue.unwrap().result_planes(), 0)
  assert_eq(
    physical.sibling
    .unwrap()
    .static_storage_bytes(element_bytes=2, result_bytes=4),
    49152L,
  )
  projection_validate_fold_resources(physical, 0)
  try projection_validate_fold_resources(physical, 4) catch {
    LimitExceeded(LaunchDimensions) => ()
    _ => fail("unexpected resource rejection")
  } noraise {
    _ => fail("dynamic memory not included")
  }
  let scheduled_shared = @luna_projection_tile_compiler.compile_projection_tile(
    request.with_fold_choices([
      @luna_projection_tile_compiler.ProjectionFoldChoice::new(
        SiblingFold,
        stages=2,
        transfer_tiles=1,
        fragment_stages=1,
        row_tiles=4,
        output_tiles=4,
        consumer_groups=16,
      ),
    ]),
  )
  let shared = projection_refine_physical(
    scheduled_shared,
    sibling_ownership=SplitSiblingPlanes,
  )
  assert_true(shared.sibling.unwrap().sibling_reuse)
  assert_eq(
    shared.sibling
    .unwrap()
    .static_storage_bytes(element_bytes=2, result_bytes=4),
    16384L,
  )
  try projection_validate_fold_resources(shared, 32772) catch {
    LimitExceeded(LaunchDimensions) => ()
    _ => fail("unexpected resource rejection")
  } noraise {
    _ => fail("shared result plane not counted")
  }
}
