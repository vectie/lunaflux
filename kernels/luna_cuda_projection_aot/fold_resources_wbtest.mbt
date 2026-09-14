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
      projection_validate_fold_resources(compilation, 16384)
    } else {
      try projection_validate_fold_resources(compilation, 16384) catch {
        LimitExceeded(LaunchDimensions) => ()
        _ => fail("unexpected resource rejection")
      } noraise {
        _ => fail("ring requiring unavailable opt-in was exported")
      }
    }
  }
}
