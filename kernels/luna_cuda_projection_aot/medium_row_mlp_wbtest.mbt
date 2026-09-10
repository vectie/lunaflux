///|
test "medium row sibling distributions retain companion and workspace contracts" {
  let parameters : ProjectionParameters = {
    family: GatedMlp,
    input_width: 1024,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 3072,
  }
  let baseline = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  for bound in [16, 32] {
    for groups in [2, 4] {
      let selected = for
        candidate in baseline
        .dispatch_plan(bound)
        .prefill()
        .output_distribution_candidates() {
        if candidate.output_tiles_per_workgroup() == groups {
          break candidate
        }
      } nobreak {
        fail("missing distribution")
      }
      let runtime = projection_build_runtime_dispatch(
        parameters,
        1024,
        projection_fixture_target(),
        [row_variant_test_record(parameters, selected)],
      )
      let profile = row_variant_test_profile()
      let primary = baseline.dispatch_plan(1024)
      let (source, variants) = projection_row_variant_sources(
        parameters,
        profile,
        runtime,
        primary,
        "mlp",
        max_registers=128,
      )
      let variant = for variant in variants {
        if variant.maximum_token_rows() == bound {
          break variant
        }
      } nobreak {
        fail("missing bound")
      }
      assert_true(source.contains("void mlp_rows\{bound}_down("))
      assert_eq(
        projection_workspace_bytes(parameters, profile, primary),
        projection_workspace_bytes(
          parameters,
          profile,
          runtime.dispatch_plan(bound),
        ),
      )
      let companion = variant.companion_dimensions().unwrap()
      let grid = variant.bounded_work_grid_x().unwrap()
      assert_true(grid <= variant.dimensions().grid_x())
      if @env.get_env_var("LUNA_TEST_EXPORT_MEDIUM_MLP") == Some("1") {
        println("LUNA_SOURCE_BEGIN mlp-g\{groups}-b\{bound}")
        println(row_variant_primary_test_source(parameters, primary))
        println(source)
        println("LUNA_SOURCE_END")
        println(
          "LUNA_LAUNCH mlp-g\{groups}-b\{bound} \{grid} \{variant.dimensions().block_x()} \{variant.dimensions().shared_memory_bytes()}",
        )
        println(
          "LUNA_DOWN mlp-g\{groups}-b\{bound} \{companion.grid_x()} \{companion.block_x()} \{companion.shared_memory_bytes()}",
        )
      }
    }
  }
}
