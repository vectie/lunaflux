///|
test "all QKV lowerings consume semantic weight partitions" {
  let parameters : ProjectionParameters = {
    family: QkvProjection,
    input_width: 128,
    output_width: 256,
    query_width: 128,
    intermediate_width: 64,
  }
  let dispatch = projection_build_runtime_dispatch(
    parameters,
    32,
    projection_fixture_target(),
    [],
  ).dispatch_plan(32)
  let compilation = projection_row_variant_compilation(
    parameters, dispatch, 1, 32,
  )
  assert_eq(
    compilation.program().concatenated_weights(),
    @luna_projection_tile_compiler.plan_projection_concatenated_rows([
      128, 64, 64,
    ]).unwrap(),
  )
  // Deliberately poison the old geometry fields after compilation. A lowerer
  // must use the bound program, not re-derive its partitions from ABI metadata.
  let stale = { ..parameters, query_width: 64, intermediate_width: 96, }
  assert_eq(
    projection_scalar_map_source("qkv", parameters, 32, 1, compilation),
    projection_scalar_map_source("qkv", stale, 32, 1, compilation),
  )
  assert_eq(
    projection_matrix_pipeline_source(parameters, compilation, 1),
    projection_matrix_pipeline_source(stale, compilation, 1),
  )
  assert_eq(
    projection_matrix_map_source(parameters, compilation),
    projection_matrix_map_source(stale, compilation),
  )
  assert_eq(
    projection_tensor_core_direct_qkv_source("qkv", parameters, 32, compilation),
    projection_tensor_core_direct_qkv_source("qkv", stale, 32, compilation),
  )
  let rebound = projection_row_variant_compilation(stale, dispatch, 1, 32)
  assert_true(compilation.program().digest() != rebound.program().digest())
  assert_true(
    projection_scalar_map_source("qkv", parameters, 32, 1, compilation) !=
    projection_scalar_map_source("qkv", stale, 32, 1, rebound),
  )
}
