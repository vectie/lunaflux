///|
test "direct sibling epilogue has unique stores and needs no reader fence" {
  let parameters = sibling_resource_test_parameters()
  let dispatch = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  ).dispatch_plan(1024)
  let compilation = @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows=32,
      maximum_tokens=1024,
      demand=AllTokenRows,
    ),
  )
  let plan = compilation.schedule().sibling_pipeline().unwrap()
  // One consumer per output tile forces the direct-register path, rather
  // than accidentally checking the separate shared-result reuse epilogue.
  let groups = plan.output_tiles()
  assert_eq(
    @luna_projection_tile_compiler.partition_projection_sibling_reuse(
      row_tiles=plan.row_width() / 16,
      output_tiles=groups,
      consumer_groups=groups,
    ),
    None,
  )
  let source = projection_sibling_fold_pipeline_source(
    plan,
    parameters,
    groups~,
  )
  assert_true(source.contains("float cg["))
  assert_false(source.contains("__syncwarp"))
  assert_true(source.contains("__syncthreads"))
  // Every lane owns distinct accumulator values and distinct global outputs.
  // There are no epilogue shared stores/reads, and the next work item does not
  // consume these outputs. Operand-ring publication/reuse fences stay intact.
  for live_rows in [1, 7, 16, 17, plan.row_width() - 1, plan.row_width()] {
    let seen = FixedArray::make(live_rows * groups * 16, 0)
    for warp in 0..<groups {
      for row in 0..<(plan.row_width() / 16) {
        for lane in 0..<32 {
          for item in 0..<8 {
            let output_row = row * 16 + lane / 4 + item % 4 / 2 * 8
            let column = warp * 16 + lane % 4 * 2 + item % 2 + item / 4 * 8
            if output_row < live_rows {
              seen[output_row * groups * 16 + column] += 1
            }
          }
        }
      }
    }
    assert_true(seen.iter().all(count => count == 1))
  }
  if @env.get_env_var("LUNA_EXPORT_REGISTER_EPILOGUE") == Some("1") {
    println("LUNA_SOURCE_BEGIN")
    println(projection_tensor_core_prelude())
    println(
      "extern \"C\" __global__ void register_epilogue(const int *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, __nv_bfloat16 *workspace) {",
    )
    println("const int token_count = counts[0];")
    println("const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;")
    println(source)
    println("}")
    println("LUNA_SOURCE_END")
    println("LUNA_GROUPS=" + groups.to_string())
  }
}
