///|
test "compiled prefill launch geometry is retained across every runtime slot" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  for threads in [64, 128, 256] {
    for partitioned in [false, true] {
      let runtime = qwen_fused_test_runtime(
        plan,
        blueprint,
        phase_specific_attention=true,
        wide_query_prefill=true,
        partitioned_prefill=partitioned,
        deep_partitioned_prefill=partitioned,
        prefill_threads=threads,
      )
      let spans = admit_reusable_qwen_fused_attention_spans(
        plan,
        @worker_protocol.ModelPlanGeneration::first(),
        blueprint,
        runtime,
      )
      assert_eq(spans[0].readonly_launch().block_x(), threads)
      guard spans[0].wide_query_prefill() is Some(wide) else {
        fail("wide slot")
      }
      assert_eq(wide.launch.block_x(), threads)
      if partitioned {
        guard spans[0].partitioned_prefill() is Some(base) &&
          spans[0].deep_partitioned_prefill() is Some(deep) else {
          fail("partition slots")
        }
        for pipeline in [base, deep] {
          assert_eq(pipeline.partial_launch.block_x(), threads)
          assert_eq(pipeline.merge_launch.block_x(), threads)
        }
      }
    }
  }
}

///|
test "compiled partial and merge still require matching worker geometry" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    partitioned_prefill=true,
    prefill_threads=128,
    merge_threads=256,
  )
  try
    ignore(
      admit_reusable_qwen_fused_attention_spans(
        plan,
        @worker_protocol.ModelPlanGeneration::first(),
        blueprint,
        runtime,
      ),
    )
  catch {
    MismatchedFusedGroupedOperation(Launch) => ()
    _ => fail("unexpected rejection")
  } noraise {
    _ => fail("mismatched merge accepted")
  }
}
