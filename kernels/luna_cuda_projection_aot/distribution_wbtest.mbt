///|
test "matrix distribution lowers through both source and launch geometry" {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=256,
    max_query_tokens=256,
    max_page_table_entries=65536,
  )
  for family in [DenseProjection, QkvProjection, GatedMlp, LanguageModelHead] {
    let parameters : ProjectionParameters = {
      family,
      input_width: 1024,
      query_width: 2048,
      output_width: if family == LanguageModelHead {
        151936
      } else if family == QkvProjection {
        4096
      } else {
        1024
      },
      intermediate_width: if family == GatedMlp {
        3072
      } else if family == QkvProjection {
        1024
      } else {
        0
      },
    }
    let baseline = projection_build_runtime_dispatch(
      parameters,
      256,
      projection_fixture_target(),
      [],
    )
    let candidates : Array[@luna_projection_strategy.ProjectionStrategy] = []
    for
      candidate in baseline
      .dispatch_plan(256)
      .prefill()
      .output_distribution_candidates() {
      candidates.push(candidate)
      if family == LanguageModelHead {
        candidates.push(
          candidate.with_selected_row_residency(
            stable_id=3000 + candidate.output_tiles_per_workgroup(),
          ),
        )
      }
    }
    for candidate in candidates {
      // This synthetic record exercises selection only; it is not a latency
      // measurement and is never installed in a serving policy.
      let record = @luna_projection_strategy.ProjectionAutotuneRecord::new(
        key=@luna_projection_strategy.ProjectionAutotuneKey::new(
          backend=projection_cuda_backend_id(),
          target=projection_cuda_target_id(projection_fixture_target()),
          numeric_type=ProjectionBFloat16,
          shape=projection_strategy_shape(parameters),
          strategy=candidate,
        ),
        median_latency_ns=1L,
        sample_count=3,
      )
      let runtime = projection_build_runtime_dispatch(
        parameters,
        256,
        projection_fixture_target(),
        [record],
      )
      let dispatch = runtime.dispatch_plan(256)
      let compilation = @luna_projection_tile_compiler.compile_projection_tile(
        @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
          dispatch~,
          maximum_rows=256,
          maximum_tokens=256,
          demand=if family == LanguageModelHead {
            QueryRowEnds
          } else {
            AllTokenRows
          },
        ),
      )
      let dimensions = projection_dimensions(profile, parameters, dispatch)
      let groups = candidate.output_tiles_per_workgroup()
      let resident = candidate.input_residency() == SelectedRowTileResidency
      assert_eq(compilation.schedule().output_tiles_per_workgroup(), groups)
      assert_eq(dimensions.block_x(), groups * 32)
      assert_eq(
        dimensions.shared_memory_bytes(),
        if family == LanguageModelHead && groups == 1 && !resident {
          9216
        } else {
          groups * 1024 +
          (if resident {
            32768
          } else if family == LanguageModelHead {
            512
          } else {
            0
          })
        },
      )
      assert_eq(compilation.schedule().selected_input_row_resident(), resident)
      let source = projection_source(
        "distribution",
        parameters,
        256,
        256,
        dispatch,
        compilation,
        false,
        true,
        max_registers=128,
      )
      assert_true(source.length() > 0)
      if (
          @env.get_env_var("LUNA_TEST_EXPORT_DISTRIBUTION") == Some("1") &&
          !resident
        ) ||
        (
          @env.get_env_var("LUNA_TEST_EXPORT_RESIDENCY") == Some("1") &&
          resident
        ) {
        println("LUNA_SOURCE_BEGIN \{projection_family_tag(family)}-\{groups}")
        println(@utf8.decode(source))
        println("LUNA_SOURCE_END")
      }
    }
  }
}
