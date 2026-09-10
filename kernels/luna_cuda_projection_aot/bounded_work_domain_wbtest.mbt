///|
test "bounded gate product publishes full independent work instead of consumer count" {
  let parameters = sibling_resource_test_parameters()
  let target = projection_fixture_target()
  let fallback = projection_build_runtime_dispatch(parameters, 1024, target, [])
  let small = sibling_resource_test_strategy(fallback, 8, 2)
  let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
    row_variant_test_record(parameters, small),
  ])
  let (_, variants) = projection_row_variant_sources(
    parameters,
    row_variant_test_profile(),
    runtime,
    runtime.dispatch_plan(1024),
    "work",
    max_registers=128,
  )
  assert_eq(variants.length(), 3)
  for i, expected in [192, 48, 48] {
    assert_eq(variants[i].bounded_work_grid_x(), Some(expected))
    assert_true(expected <= variants[i].dimensions().grid_x())
  }
  assert_eq(variants[0].maximum_token_rows(), 8)
  assert_eq(variants[1].maximum_token_rows(), 16)
  assert_eq(variants[2].maximum_token_rows(), 32)
}

///|
test "bounded work metadata is shared by dense and segmented matrix pipelines" {
  for family in [DenseProjection, QkvProjection] {
    let parameters : ProjectionParameters = {
      family,
      input_width: 1024,
      output_width: 4096,
      query_width: if family == QkvProjection {
        2048
      } else {
        0
      },
      intermediate_width: if family == QkvProjection {
        1024
      } else {
        0
      },
    }
    let runtime = projection_build_runtime_dispatch(
      parameters,
      1024,
      projection_fixture_target(),
      [],
    )
    let (_, variants) = projection_row_variant_sources(
      parameters,
      row_variant_test_profile(),
      runtime,
      runtime.dispatch_plan(1024),
      "matrix_work",
      max_registers=128,
    )
    assert_false(variants.is_empty())
    for variant in variants {
      assert_true(variant.bounded_work_grid_x() is Some(_))
    }
  }
}
