///|
test "startup resolves head variants and both MLP companion symbols" {
  let base = qwen_fused_test_blueprint(qwen_fused_test_plan())
  for explicit_companion in [false, true] {
    for kind in [@plan.OperationKind::LanguageModelHead, GatedMlp] {
      let width = if kind == LanguageModelHead { 151936 } else { 1024 }
      let dimensions = @launch_contract.AotLaunchDimensions::new(
        grid_x=if kind == LanguageModelHead { 1187 } else { 96 },
        grid_y=1,
        grid_z=1,
        block_x=if kind == LanguageModelHead { 256 } else { 64 },
        block_y=1,
        block_z=1,
        shared_memory_bytes=if kind == LanguageModelHead { 40960 } else { 2048 },
      )
      let operands : Array[PagedKvOperandBinding] = [
        {
          role: ActivationOutput(@plan.ActivationRef::new(0)),
          space: ActivationWorkspaceArena,
          offset_bytes: 0L,
          byte_count: 1024L * width.to_int64() * 2L,
          alignment: 16L,
        },
      ]
      if kind == GatedMlp {
        operands.push({
          role: Workspace,
          space: ActivationWorkspaceArena,
          offset_bytes: 0L,
          byte_count: 1024L * 3072L * 2L,
          alignment: 16L,
        })
      }
      let companion_dimensions = if explicit_companion && kind == GatedMlp {
        Some(
          @launch_contract.AotLaunchDimensions::new(
            grid_x=1024,
            grid_y=1,
            grid_z=1,
            block_x=128,
            block_y=1,
            block_z=1,
            shared_memory_bytes=4096,
          ),
        )
      } else {
        None
      }
      let step = {
        ..base.steps()[0],
        operation_kind: kind,
        function_symbol: "matrix",
        dimensions,
        companion_dimensions,
        operands: ReadOnlyArray::from_array(operands),
        row_variants: [
          @artifact.KernelRowVariant::new(
            maximum_token_rows=8,
            function_symbol="matrix_rows8",
            dimensions~,
            companion_dimensions~,
          ),
        ],
      }
      let blueprint = {
        ..base,
        profile: @launch_contract.PagedKernelProfile::new(
          id=base.profile().id(),
          max_query_rows=32,
          max_query_tokens=1024,
          max_page_table_entries=base.profile().max_page_table_entries(),
        ),
        steps: [step],
      }
      let (functions, indices, companions) = collect_paged_functions(blueprint)
      let resources = empty_paged_resources()
      prepare_projection_variants(blueprint, functions, resources)
      let chosen = select_projection_variant(
        resources.projection_variants,
        indices[0],
        8,
      ).unwrap()
      assert_eq(functions[chosen.function_index].symbol, "matrix_rows8")
      assert_eq(
        paged_bucket_launch(chosen.launch, 8).grid_x(),
        dimensions.grid_x(),
      )
      assert_true(
        select_projection_variant(resources.projection_variants, indices[0], 1)
        is None,
      )
      assert_true(
        select_projection_variant(resources.projection_variants, indices[0], 9)
        is None,
      )
      if kind == GatedMlp {
        let down = select_projection_variant(
          resources.projection_variants,
          companions[0],
          8,
        ).unwrap()
        assert_eq(functions[down.function_index].symbol, "matrix_rows8_down")
        assert_eq(
          paged_bucket_launch(down.launch, 8).grid_x(),
          if explicit_companion {
            16
          } else {
            32
          },
        )
        assert_eq(
          paged_bucket_launch(down.launch, 8).block_x(),
          if explicit_companion {
            128
          } else {
            64
          },
        )
        assert_eq(
          paged_bucket_launch(down.launch, 8).shared_memory_bytes(),
          if explicit_companion {
            4096
          } else {
            2048
          },
        )
        assert_eq(resources.projection_variants.length(), 2)
      } else {
        assert_eq(resources.projection_variants.length(), 1)
        assert_eq(
          paged_bucket_launch(chosen.launch, 8).shared_memory_bytes(),
          40960,
        )
      }
    }
  }
}
