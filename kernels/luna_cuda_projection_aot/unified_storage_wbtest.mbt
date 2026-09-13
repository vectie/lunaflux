///|
test "unified folds cover every MLP output distribution at the small matrix boundary" {
  for width in [256, 512] {
    let parameters : ProjectionParameters = {
      family: GatedMlp,
      input_width: width,
      output_width: 256,
      query_width: 0,
      intermediate_width: 256,
    }
    let runtime = projection_build_runtime_dispatch(
      parameters,
      1024,
      projection_fixture_target(),
      [],
    )
    for
      candidate in runtime
      .dispatch_plan(256)
      .prefill()
      .output_distribution_candidates() {
      let selected = projection_build_runtime_dispatch(
        parameters,
        1024,
        projection_fixture_target(),
        [row_variant_test_record(parameters, candidate)],
      )
      let dispatch = selected.dispatch_plan(256)
      let source = row_variant_primary_test_source(
        parameters,
        dispatch,
        maximum_tokens=256,
      )
      assert_false(source.contains("wmma::"))
      assert_true(source.contains("cp.async.commit_group"))
      assert_true(source.contains("mma.sync.aligned.m16n8k16"))
      assert_true(source.contains("output_row < token_count"))
      if @env.get_env_var("LUNA_TEST_EXPORT_UNIFIED_STORAGE") == Some("1") {
        println(
          "LUNA_SOURCE_BEGIN mlp-\{width}-g\{candidate.output_tiles_per_workgroup()}",
        )
        println(source)
        println("LUNA_SOURCE_END")
      }
    }
  }
}

///|
test "strip and resident schedules share the matrix fold with masked columns" {
  let parameters : ProjectionParameters = {
    family: LanguageModelHead,
    input_width: 1024,
    output_width: 80,
    query_width: 0,
    intermediate_width: 0,
  }
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  for
    candidate in runtime
    .dispatch_plan(256)
    .prefill()
    .output_distribution_candidates() {
    if candidate.output_tiles_per_workgroup() != 1 &&
      candidate.output_tiles_per_workgroup() != 8 {
      continue
    }
    for resident in [false, true] {
      let strategy = if resident {
        candidate.with_selected_row_residency(
          stable_id=3000 + candidate.output_tiles_per_workgroup(),
        )
      } else {
        candidate
      }
      let selected = projection_build_runtime_dispatch(
        parameters,
        1024,
        projection_fixture_target(),
        [row_variant_test_record(parameters, strategy)],
      )
      let source = row_variant_primary_test_source(
        parameters,
        selected.dispatch_plan(256),
        maximum_tokens=256,
      )
      assert_false(source.contains("wmma::"))
      assert_true(source.contains("mma.sync.aligned.m16n8k16"))
      if @env.get_env_var("LUNA_TEST_EXPORT_UNIFIED_STORAGE") == Some("1") {
        println(
          "LUNA_SOURCE_BEGIN head-\{if resident { "resident" } else { "strip" }}-g\{candidate.output_tiles_per_workgroup()}",
        )
        println(source)
        println("LUNA_SOURCE_END")
      }
    }
  }
}

///|
test "ordered and tree folds share arithmetic without implicit reassociation" {
  let dots : Array[CudaProjectionDot] = [
    { accumulator: "value", left: "x[inner]", right: "w[inner]", },
  ]
  let ordered = projection_dot_fold_source(33, dots[:], OrderedDot)
  let tree = projection_dot_fold_source(33, dots[:], StridedPairwiseDot(32))
  assert_true(ordered.contains("inner = 0; inner < 33; inner += 1"))
  assert_false(ordered.contains("shfl"))
  assert_true(tree.contains("inner = lane; inner < 33; inner += 32"))
  assert_true(tree.contains("delta = 16"))
  for source in [ordered, tree] {
    assert_true(
      source.contains("__fadd_rn(value, __fmul_rn(x[inner], w[inner]))"),
    )
    assert_false(source.contains("__fmaf"))
  }
}

///|
test "scalar sibling materialization preserves the same ordered F32 product" {
  let parameters : ProjectionParameters = {
    family: GatedMlp,
    input_width: 4,
    output_width: 4,
    query_width: 0,
    intermediate_width: 8,
  }
  for materialize in [false, true] {
    let source = @utf8.decode(
      projection_scalar_mlp_source("baseline", parameters, 4, materialize),
    )
    assert_true(source.contains("gate = __fadd_rn(gate, __fmul_rn("))
    assert_true(source.contains("up = __fadd_rn(up, __fmul_rn("))
    assert_true(source.contains("const float product = __fmul_rn(silu, up)"))
    assert_eq(
      source.contains("extern __shared__ float combined[]"),
      materialize,
    )
    assert_false(source.contains("__float2bfloat16_rn(product)"))
    if !materialize &&
      @env.get_env_var("LUNA_TEST_EXPORT_UNIFIED_STORAGE") == Some("1") {
      println("LUNA_SOURCE_BEGIN scalar-mlp-inline")
      println(source)
      println("LUNA_SOURCE_END")
    }
  }
}
