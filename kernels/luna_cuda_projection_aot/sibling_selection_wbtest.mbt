///|
test "measured sibling ownership selects regenerated executable and bounded variants" {
  let input = @plan.ActivationRef::new(0)
  let tensors = [0, 1, 2].map(i => @plan.TensorRef::new(i))
  let operation = projection_fixture_operation(
    1,
    GatedMlp,
    [
      Activation(input),
      Tensor(tensors[0]),
      Tensor(tensors[1]),
      Tensor(tensors[2]),
    ],
    GatedMlpShape(hidden_size=1024, intermediate_size=3072),
    @plan.KernelCapabilityId::gated_mlp(),
  )
  let profile = row_variant_test_profile()
  let operands = [
    projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
    projection_fixture_operand(ActivationInput(input), 1024L * 1024L * 2L, 16L),
    projection_fixture_operand(WeightInput(tensors[0]), 3072L * 1024L * 2L, 16L),
    projection_fixture_operand(WeightInput(tensors[1]), 3072L * 1024L * 2L, 16L),
    projection_fixture_operand(WeightInput(tensors[2]), 3072L * 1024L * 2L, 16L),
    projection_fixture_operand(
      ActivationOutput(@plan.ActivationRef::new(1)),
      1024L * 1024L * 2L,
      16L,
    ),
    projection_fixture_operand(Workspace, 1024L * 3072L * 2L, 16L),
  ]
  let target = projection_fixture_target()
  let compiler = projection_fixture_compiler()
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
  let base = baseline.source_digest().as_hex()
  for
    (token, ownership, transport) in [
      (
        "retained",
        @luna_projection_tile_compiler.RetainSiblingProducer,
        @luna_projection_tile_compiler.BaselineSiblingTransport,
      ),
      ("coowned", CoownedSiblingValues, BaselineSiblingTransport),
      ("coowned", CoownedSiblingValues, SegmentedSiblingTransport),
    ] {
    let choices = if token == "coowned" {
      [
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          SiblingFold,
          stages=2,
          transfer_tiles=2,
          fragment_stages=1,
          row_tiles=4,
          output_tiles=8,
          consumer_groups=8,
        ),
      ]
    } else {
      []
    }
    let encoded = (if choices.is_empty() {
        ""
      } else {
        "sibling:2:2:1:4:8:0:8,"
      }) +
      "epilogue:\{token}" +
      (if transport == SegmentedSiblingTransport {
        ",transport:segmented"
      } else {
        ""
      })
    let alternative = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
      operation,
      profile,
      target,
      compiler,
      operands,
      runtime,
      limits,
      sibling_ownership=ownership,
      sibling_transport=transport,
      fold_choices=choices,
    )
    let artifact = alternative.source_digest().as_hex()
    assert_false(base == artifact)
    assert_eq(alternative.launch_dimensions().shared_memory_bytes(), 0)
    assert_eq(
      alternative.launch_dimensions().block_x(),
      if token == "coowned" {
        256
      } else {
        512
      },
    )
    assert_true(
      @utf8.decode(alternative.recipe_bytes()).contains(
        "sibling_epilogue_ownership=",
      ),
    )
    for latency in [50, 99, 150] {
      let records = @luna_projection_tuning.parse_projection_fold_records(
        @utf8.encode(
          "luna-projection-fold-v2\nscope\ttest\nrecord\t\{base}\t\{base}\t100\t3\t-\nrecord\t\{base}\t\{artifact}\t\{latency}\t3\t\{encoded}\n",
        ),
        scope="test",
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
        minimum_improvement_basis_points=200,
      )
      assert_eq(
        selected.source_digest().as_hex(),
        if latency == 50 {
          artifact
        } else {
          base
        },
      )
    }
    // Source binding includes bounded variants, not only the main kernel.
    let source = @utf8.decode(alternative.source_bytes())
    assert_true(alternative.row_variants.length() > 0)
    for variant in alternative.row_variants {
      let body = source.split("void \{variant.function_symbol()}(").to_array()[1]
        .split("#include")
        .to_array()[0]
      assert_false(body.contains("float *gate_tiles"))
      assert_false(body.contains("sibling_output_bytes"))
      assert_eq(variant.dimensions().shared_memory_bytes(), 0)
      if token == "coowned" {
        assert_eq(variant.dimensions().block_x(), 256)
        assert_true(variant.dimensions().grid_x() >= 384)
        assert_true(variant.bounded_work_grid_x().unwrap() > 0)
        if variant.maximum_token_rows() <= 16 {
          assert_true(variant.bounded_work_grid_x().unwrap() < 384)
        }
        if transport == SegmentedSiblingTransport {
          assert_true(body.contains("transfer_sources["))
          assert_false(body.contains("bool overlap"))
        }
      }
    }
    let stale = @luna_projection_tuning.parse_projection_fold_records(
      @utf8.encode(
        "luna-projection-fold-v2\nscope\ttest\nrecord\t\{base}\t\{base}\t100\t3\t-\nrecord\t\{base}\t\{"0".repeat(64)}\t50\t3\t\{encoded}\n",
      ),
      scope="test",
    )
    try
      lower_measured_projection_cuda_aot_candidate(
        operation,
        profile,
        target,
        compiler,
        operands,
        runtime,
        limits,
        stale[:],
      )
    catch {
      Invalid(CompilerPolicy) => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("stale ownership executable accepted")
    }
  }
}
