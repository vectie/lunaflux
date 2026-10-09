///|
test "partition counts boundary weights only on their owning host" {
  let plan = @decoder_pipeline_plan.DecoderPipelinePlan::partition(
    [4L, 4L, 4L, 4L, 4L],
    [15L, 16L],
    stage_overhead_bytes=1L,
    ingress_bytes=2L,
    egress_bytes=3L,
  )
  assert_eq(plan.stages()[0].layer_range(), (0, 3))
  assert_eq(plan.stages()[1].layer_range(), (3, 5))
  assert_eq(plan.stage_bytes().to_owned(), [15L, 12L])
  assert_true(plan.stages()[0].text_boundary() is Ingress)
  assert_true(plan.stages()[1].text_boundary() is Egress)
}

///|
test "placement preserves a feasible suffix on heterogeneous hosts" {
  let plan = @decoder_pipeline_plan.DecoderPipelinePlan::partition(
    [2L, 6L, 1L, 1L],
    [10L, 2L, 2L],
    stage_overhead_bytes=1L,
    ingress_bytes=0L,
    egress_bytes=0L,
  )
  assert_eq(plan.stages()[0].layer_range(), (0, 2))
  assert_eq(plan.stages()[1].layer_range(), (2, 3))
  assert_eq(plan.stages()[2].layer_range(), (3, 4))
  assert_true(plan.stages()[1].text_boundary() is Interior)
}

///|
test "single host owns both ends and caller mutation cannot change placement" {
  let sizes = [3L, 5L]
  let budgets = [20L]
  let plan = @decoder_pipeline_plan.DecoderPipelinePlan::partition(
    sizes,
    budgets,
    stage_overhead_bytes=1L,
    ingress_bytes=2L,
    egress_bytes=4L,
  )
  sizes[0] = 100L
  budgets[0] = 1L
  assert_eq(plan.stage_bytes()[0], 15L)
  assert_true(plan.stages()[0].text_boundary() is Complete)
}

///|
test "memory rejection does not conceal an oversized layer or overflow" {
  for sizes in [[11L, 1L], [-1L, 1L], [9223372036854775807L, 1L]] {
    try
      @decoder_pipeline_plan.DecoderPipelinePlan::partition(
        sizes,
        [10L, 10L],
        stage_overhead_bytes=0L,
        ingress_bytes=0L,
        egress_bytes=0L,
      )
    catch {
      @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
      e => raise e
    } noraise {
      _ => fail("invalid footprint accepted")
    }
  }
}
