///|
fn sibling_resource_test_parameters(
  family? : ProjectionAotFamily = GatedMlp,
) -> ProjectionParameters {
  {
    family,
    input_width: 1024,
    output_width: 1024,
    query_width: 0,
    intermediate_width: if family == GatedMlp {
      3072
    } else {
      0
    },
  }
}

///|
fn sibling_resource_test_strategy(
  runtime : @luna_projection_strategy.ProjectionRuntimeDispatch,
  rows : Int,
  groups : Int,
) -> @luna_projection_strategy.ProjectionStrategy raise {
  for
    strategy in runtime
    .dispatch_plan(rows)
    .prefill()
    .output_distribution_candidates() {
    if strategy.output_tiles_per_workgroup() == groups {
      return strategy
    }
  }
  fail("missing resource test distribution")
}

///|
fn sibling_resource_test_compilation(
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
) -> @luna_projection_tile_compiler.ScheduledProjectionTileCompilation raise {
  @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows=32,
      maximum_tokens=1024,
      demand=AllTokenRows,
      single_row_numerics=AllowDeterministicDotTree,
    ),
  )
}

///|
test "sibling launch bounds preserve caller budget and companion entry point" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let dispatch = runtime.dispatch_plan(1024)
  let compilation = sibling_resource_test_compilation(dispatch)
  let schedule = compilation.schedule()
  assert_eq(schedule.output_tiles_per_workgroup(), 16)
  assert_true(schedule.sibling_pipeline() is Some(_))
  for max_registers in [32, 63, 64, 128] {
    let qualifier = projection_sibling_launch_bounds_source(
      schedule,
      max_registers~,
    )
    assert_eq(
      qualifier,
      if max_registers >= 64 {
        "__launch_bounds__(512, 2) "
      } else {
        ""
      },
    )
    let source = @utf8.decode(
      projection_source(
        "resource_gate",
        parameters,
        1024,
        32,
        dispatch,
        compilation,
        false,
        true,
        max_registers~,
      ),
    )
    assert_true(
      source.contains(
        "extern \"C\" __global__ " + qualifier + "void resource_gate(",
      ),
    )
    assert_true(
      source.contains("extern \"C\" __global__ void resource_gate_down("),
    )
    assert_eq(source.contains("__launch_bounds__"), max_registers >= 64)
    assert_false(source.contains("__maxnreg__"))
  }
}

///|
test "sibling resource target does not extrapolate to smaller workgroups" {
  let parameters = sibling_resource_test_parameters()
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  for groups in [1, 2, 4, 8] {
    let strategy = sibling_resource_test_strategy(baseline, 1024, groups)
    let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
      row_variant_test_record(parameters, strategy),
    ])
    let dispatch = runtime.dispatch_plan(1024)
    let compilation = sibling_resource_test_compilation(dispatch)
    let schedule = compilation.schedule()
    assert_eq(schedule.output_tiles_per_workgroup(), groups)
    let sibling = schedule.sibling_pipeline().unwrap()
    let reuse = @luna_projection_tile_compiler.partition_projection_sibling_reuse(
      row_tiles=sibling.row_width() / 16,
      output_tiles=sibling.output_tiles(),
      consumer_groups=groups,
    )
    assert_eq(reuse is Some(_), groups >= 4)
    for max_registers in [32, 64, 128] {
      assert_eq(
        projection_sibling_launch_bounds_source(schedule, max_registers~),
        "",
      )
      let source = @utf8.decode(
        projection_source(
          "smaller_gate",
          parameters,
          1024,
          32,
          dispatch,
          compilation,
          false,
          true,
          max_registers~,
        ),
      )
      assert_false(source.contains("__launch_bounds__"))
    }
  }
}

///|
test "a same-width matrix without sibling reuse has no sibling resource target" {
  let parameters = sibling_resource_test_parameters(family=DenseProjection)
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  let strategy = sibling_resource_test_strategy(baseline, 1024, 16)
  let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
    row_variant_test_record(parameters, strategy),
  ])
  let dispatch = runtime.dispatch_plan(1024)
  let compilation = sibling_resource_test_compilation(dispatch)
  let schedule = compilation.schedule()
  assert_eq(schedule.output_tiles_per_workgroup(), 16)
  assert_true(schedule.sibling_pipeline() is None)
  assert_eq(
    projection_sibling_launch_bounds_source(schedule, max_registers=128),
    "",
  )
  let source = @utf8.decode(
    projection_source(
      "matrix_only",
      parameters,
      1024,
      32,
      dispatch,
      compilation,
      false,
      true,
      max_registers=128,
    ),
  )
  assert_false(source.contains("__launch_bounds__"))
}

///|
test "measured row variants inherit the caller register budget independently" {
  let parameters = sibling_resource_test_parameters()
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  let primary_strategy = sibling_resource_test_strategy(baseline, 1024, 8)
  let variant_strategy = sibling_resource_test_strategy(baseline, 8, 16)
  let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
    row_variant_test_record(parameters, primary_strategy),
    row_variant_test_record(parameters, variant_strategy),
  ])
  let primary = runtime.dispatch_plan(1024)
  assert_eq(primary.prefill().output_tiles_per_workgroup(), 8)
  assert_eq(runtime.dispatch_plan(8).prefill().output_tiles_per_workgroup(), 16)
  let profile = row_variant_test_profile()
  for max_registers in [32, 63, 64, 128] {
    let (source, variants) = projection_row_variant_sources(
      parameters,
      profile,
      runtime,
      primary,
      "variant_gate",
      max_registers~,
    )
    assert_eq(variants.length(), 1)
    assert_eq(variants[0].maximum_token_rows(), 8)
    assert_eq(variants[0].dimensions().block_x(), 512)
    let qualifier = if max_registers >= 64 {
      "__launch_bounds__(512, 2) "
    } else {
      ""
    }
    assert_true(
      source.contains(
        "extern \"C\" __global__ " + qualifier + "void variant_gate_rows8(",
      ),
    )
    assert_true(
      source.contains("extern \"C\" __global__ void variant_gate_rows8_down("),
    )
    assert_eq(source.contains("__launch_bounds__"), max_registers >= 64)
  }
}
