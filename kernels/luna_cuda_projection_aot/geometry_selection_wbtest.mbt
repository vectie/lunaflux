///|
/// Synthetic selector fixtures, deliberately not physical timing records.
/// The physical 13.0.88/sm121 flags and outcomes belong to the benchmark report.
test "source bound output and down reuse records reach executable launch plans" {
  for mlp in [false, true] {
    let input = @plan.ActivationRef::new(0)
    let tensors = [0, 1, 2].map(i => @plan.TensorRef::new(i))
    let inputs : Array[@plan.ValueRef] = [Activation(input), Tensor(tensors[0])]
    if mlp {
      inputs.push(Tensor(tensors[1]))
      inputs.push(Tensor(tensors[2]))
    }
    let operation = projection_fixture_operation(
      1,
      if mlp {
        GatedMlp
      } else {
        OutputProjection
      },
      inputs,
      if mlp {
        GatedMlpShape(hidden_size=1024, intermediate_size=3072)
      } else {
        ProjectionShape(input_width=2048, output_width=1024)
      },
      if mlp {
        @plan.KernelCapabilityId::gated_mlp()
      } else {
        @plan.KernelCapabilityId::dense_projection()
      },
    )
    let profile = @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(1),
      max_query_rows=32,
      max_query_tokens=2048,
      max_page_table_entries=8192,
    )
    let width = if mlp { 1024L } else { 2048L }
    let operands = [
      projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
      projection_fixture_operand(
        ActivationInput(input),
        2048L * width * 2L,
        16L,
      ),
      projection_fixture_operand(
        WeightInput(tensors[0]),
        if mlp {
          3072L * 1024L * 2L
        } else {
          1024L * 2048L * 2L
        },
        16L,
      ),
    ]
    if mlp {
      operands.push(
        projection_fixture_operand(
          WeightInput(tensors[1]),
          3072L * 1024L * 2L,
          16L,
        ),
      )
      operands.push(
        projection_fixture_operand(
          WeightInput(tensors[2]),
          3072L * 1024L * 2L,
          16L,
        ),
      )
    }
    operands.push(
      projection_fixture_operand(
        ActivationOutput(@plan.ActivationRef::new(1)),
        2048L * 1024L * 2L,
        16L,
      ),
    )
    if mlp {
      operands.push(
        projection_fixture_operand(Workspace, 2048L * 3072L * 2L, 16L),
      )
    }
    let target = projection_fixture_target()
    let compiler = @luna_specialization.LunaCompilerPolicy::new(
      identity=@luna_specialization.LunaCompilerIdentity::new(
        toolchain_digest_sha256="0".repeat(64),
        major=13,
        minor=0,
        patch=88,
      ),
      optimization_level=3,
      fused_multiply_add=false,
      reassociate=false,
      max_registers=255,
    )
    let limits = projection_fixture_limits(max_source_bytes=1048576)
    let runtime = prepare_projection_cuda_runtime_dispatch(
      operation,
      profile,
      target,
      operands,
      [],
    )
    let baseline = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
      operation, profile, target, compiler, operands, runtime, limits,
    )
    let choice = @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      if mlp {
        IntermediateFold
      } else {
        MatrixFold
      },
      stages=2,
      transfer_tiles=4,
      fragment_stages=1,
      row_tiles=if mlp { 8 } else { 4 },
      output_tiles?=if mlp { None } else { Some(4) },
      hoist_addresses=true,
    )
    let small = @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      if mlp {
        SiblingFold
      } else {
        MatrixFold
      },
      stages=2,
      transfer_tiles=4,
      fragment_stages=1,
      row_tiles=1,
      output_tiles=2,
      consumer_groups=2,
      maximum_token_rows=16,
      hoist_addresses=!mlp,
    )
    let alternative = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
      operation,
      profile,
      target,
      compiler,
      operands,
      runtime,
      limits,
      fold_choices=[choice, small],
    )
    let base = baseline.source_digest().as_hex()
    let artifact = alternative.source_digest().as_hex()
    let encoded = if mlp {
      "intermediate:2:4:1:8:0:1,sibling:2:4:1:1:2:0:2:16"
    } else {
      "matrix:2:4:1:4:4:1,matrix:2:4:1:1:2:1:2:16"
    }
    let records = @luna_projection_tuning.parse_projection_fold_records(
      @utf8.encode(
        "luna-projection-fold-v3\nscope\tdiagnostic-selector-fixture\nrecord\t\{base}\t\{base}\t100000\t3\t-\nrecord\t\{base}\t\{artifact}\t70000\t3\t\{encoded}\n",
      ),
      scope="diagnostic-selector-fixture",
    )
    let selected = lower_measured_projection_cuda_aot_candidate(
      operation,
      profile,
      target,
      compiler,
      operands,
      runtime,
      limits,
      records[:],
    )
    assert_eq(selected.source_digest(), alternative.source_digest())
    assert_eq(selected.launch_dimensions(), alternative.launch_dimensions())
    for variant in selected.row_variants {
      if variant.maximum_token_rows() <= 16 {
        assert_eq(variant.dimensions().block_x(), 64)
      }
    }
    assert_true(
      @utf8.decode(selected.recipe_bytes()).contains("max_registers=255"),
    )
    assert_true(
      @utf8.decode(selected.recipe_bytes()).contains("compiler_version=13.0.88"),
    )
    if mlp {
      let dimensions = selected.companion_dimensions().unwrap()
      assert_eq(dimensions.grid_x(), 256)
      assert_eq(dimensions.block_x(), 128)
      assert_true(
        dimensions.grid_x() < baseline.companion_dimensions().unwrap().grid_x(),
      )
      assert_eq(
        selected
        .tile_compilation()
        .schedule()
        .intermediate_pipeline()
        .unwrap()
        .row_width(),
        128,
      )
    } else {
      assert_eq(selected.launch_dimensions().grid_x(), 512)
      assert_eq(selected.launch_dimensions().block_x(), 256)
      assert_true(
        selected.launch_dimensions().grid_x() <
        baseline.launch_dimensions().grid_x(),
      )
      assert_eq(
        selected
        .tile_compilation()
        .schedule()
        .matrix_pipeline()
        .unwrap()
        .row_width(),
        64,
      )
    }
    let source = @utf8.decode(selected.source_bytes())
    assert_true(source.contains("input_address_base"))
    assert_true(source.contains("cp.async"))
    assert_true(selected.row_variants().length() > 0)
    for variant in selected.row_variants() {
      let body = source.split("void \{variant.function_symbol()}(").to_array()[1]
        .split("#include")
        .to_array()[0]
      assert_true(body.contains("input_address_base"))
      if mlp {
        assert_true(
          variant.companion_dimensions().unwrap().grid_x() <=
          dimensions_of_down_bound(variant.maximum_token_rows()),
        )
      }
    }
    let unmeasured = lower_measured_projection_cuda_aot_candidate(
      operation,
      profile,
      target,
      compiler,
      operands,
      runtime,
      limits,
      [],
    )
    assert_eq(unmeasured.source_digest(), baseline.source_digest())
  }
}

///|
fn dimensions_of_down_bound(rows : Int) -> Int {
  Int::max((rows + 127) / 128 * 16, 64)
}
