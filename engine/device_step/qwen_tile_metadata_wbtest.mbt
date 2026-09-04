///|
test "partition tile semantics survive independent launch envelopes" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  for tile in [16, 32, 64] {
    for grid in [1, 2] {
      let runtime = qwen_fused_test_runtime(
        plan,
        blueprint,
        phase_specific_attention=true,
        full_ingress=true,
        partitioned_prefill=true,
        partitioned_prefill_grid_x=grid,
        partitioned_query_tile_rows=tile,
        merge_query_tile_rows=tile,
      )
      let spans = admit_reusable_qwen_fused_attention_spans(
        plan,
        @worker_protocol.ModelPlanGeneration::first(),
        blueprint,
        runtime,
      )
      guard spans[0].partitioned_prefill() is Some(pipeline) else {
        fail("missing partition plan")
      }
      assert_eq(pipeline.partial_launch.grid_x(), grid)
      assert_eq(pipeline.query_tile_rows, tile)
      guard qwen_partitioned_bucket_launches(pipeline)
        is (
          QueryTileCappedGridX(partial, partial_tile),
          QueryTileCappedGridX(merge, merge_tile),
        ) else {
        fail("partition launch lost semantic tiling")
      }
      assert_eq(partial.grid_x(), grid)
      assert_eq(merge.grid_x(), grid)
      assert_eq(partial_tile, tile)
      assert_eq(merge_tile, tile)
      assert_eq(qwen_partitioned_prefill_query_tile(blueprint, spans), tile)
    }
  }
}

///|
test "partition partial and merge must share semantic tile size" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    full_ingress=true,
    partitioned_prefill=true,
    partitioned_query_tile_rows=32,
    merge_query_tile_rows=64,
  )
  try
    admit_reusable_qwen_fused_attention_spans(
      plan,
      @worker_protocol.ModelPlanGeneration::first(),
      blueprint,
      runtime,
    )
  catch {
    MismatchedFusedGroupedOperation(Launch) => ()
    _ => fail("unexpected tile mismatch error")
  } noraise {
    _ => fail("inconsistent partial/merge tile accepted")
  }
}
