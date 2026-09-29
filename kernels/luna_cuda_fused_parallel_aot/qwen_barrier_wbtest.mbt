///|
test "ingress tail tiles keep inactive column owners in block barriers" {
  for dimension in [16, 32, 64, 128] {
    let candidate = qwen_fused_test_candidate(
      head_dimension=dimension,
      maximum_tokens=32,
    )
    let production = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
      candidate,
      fused_test_limits(),
    )
    let source = @utf8.decode(production.source_bytes())
    let map = @luna_projection_tile_compiler.ProjectionVectorTransferMap::new(
      vectors=dimension / 16,
      owners=4,
    ).unwrap()
    let (_, _, rounds) = map.geometry()
    assert_true(source.contains("#define LF_COLUMN_ROUNDS \{rounds}\n"))
    assert_true(source.contains("window < LF_COLUMN_WINDOWS"))
    assert_true(source.contains("slot < LF_LIVE_COLUMNS"))
    assert_false(source.contains("column_base < LF_HEAD_DIM; column_base +="))
    assert_true(source.contains("cp.async.wait_group 0;"))
    assert_true(source.contains("__syncthreads();"))
    assert_true(source.contains("if (column_base < LF_HEAD_DIM)"))
    if @env.get_env_var("LUNA_TEST_EXPORT_INGRESS_BARRIERS") == Some("1") {
      println("LUNA_SOURCE_BEGIN ingress-\{dimension}")
      println(source)
      println("LUNA_SOURCE_END")
    }
  }
}

///|
test "ingress physical IR bounds accumulators and shares the ordered fold input" {
  let candidate = qwen_fused_test_candidate(
    head_dimension=128,
    maximum_tokens=32,
  )
  let plan = qwen_ingress_column_plan(
    candidate.tile_compilation(),
    policy=candidate.ingress_policy(),
  )
  assert_eq(plan.geometry(), (4, 2, 2, 1))
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("float> c[1][LF_LIVE_COLUMNS]"))
  assert_true(
    source.contains(
      "mma_sync(c[row_fragment][column_slot], a[fragment_stage][row_fragment], b[fragment_stage][column_slot], c[row_fragment][column_slot])",
    ),
  )
  let pipeline = candidate
    .tile_compilation()
    .schedule()
    .matrix_pipeline()
    .unwrap()
  assert_eq(pipeline.row_width(), 16)
  assert_eq(pipeline.output_tiles(), 8)
  // Both operand producers use the same finite ring as ordinary projections.
  // Publication is outside the independent consumer map, including tail CTAs.
  assert_true(source.contains("cp.async.cg.shared.global"))
  assert_true(
    source.contains(
      "current + row_fragment * 16 * \{pipeline.transfer_width()} + fragment_inner, \{pipeline.transfer_width()}",
    ),
  )
  assert_true(
    source.contains(
      "for (int inner = 0; inner < 512; inner += \{pipeline.transfer_width()})",
    ),
  )
  assert_true(
    source.contains("inner = lane; inner < LF_INPUT_WIDTH; inner += 32"),
  )
  assert_true(
    @utf8.decode(candidate.recipe_bytes()).contains(
      "column_fold_plan_sha256=" +
      fused_parallel_digest(plan.canonical_bytes()).as_hex(),
    ),
  )
}

///|
test "ingress consumes offline multistage and fragment choices without changing semantics" {
  let baseline = qwen_fused_test_candidate(head_dimension=128)
  for stages in [2, 3, 4] {
    let candidate = qwen_fused_test_candidate(head_dimension=128, fold_choices=[
      @luna_projection_tile_compiler.ProjectionFoldChoice::new(
        MatrixFold,
        stages~,
        transfer_tiles=2,
        fragment_stages=2,
      ),
    ])
    let pipeline = candidate
      .tile_compilation()
      .schedule()
      .matrix_pipeline()
      .unwrap()
    assert_eq(pipeline.stage_count(), stages)
    assert_eq(pipeline.fragment_stage_count(), 2)
    assert_eq(pipeline.transfer_width(), 32)
    assert_eq(
      candidate.tile_compilation().program().digest(),
      baseline.tile_compilation().program().digest(),
    )
    assert_true(
      @utf8.decode(candidate.source_bytes()).contains("fragment_stage ^ 1"),
    )
    assert_true(
      @utf8.decode(candidate.source_bytes()).contains(
        "cp.async.wait_group \{stages - 2};",
      ),
    )
  }
  try
    qwen_fused_test_candidate(head_dimension=128, fold_choices=[
      @luna_projection_tile_compiler.ProjectionFoldChoice::new(
        MatrixFold,
        stages=4,
        transfer_tiles=4,
        fragment_stages=1,
      ),
    ])
  catch {
    FusedParallelAotError::FusedParallelUnsupported(StaticResources) => ()
    _ => fail("wrong oversized ring error")
  } noraise {
    _ => fail("static shared-memory overflow accepted")
  }
}

///|
test "unsupported ingress matrix width returns an error before source emission" {
  try qwen_fused_test_candidate(head_dimension=8) catch {
    FusedParallelAotError::FusedParallelUnsupported(StaticResources) => ()
    _ => fail("unexpected unsupported-width error")
  } noraise {
    _ => fail("sub-matrix head width accepted")
  }
}
