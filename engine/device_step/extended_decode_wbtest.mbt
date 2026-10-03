///|
test "owned KV64 and matrix decode geometry reaches every attention span" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  for
    law in [
      "owned4-blockwise-fma-f32-probability-v4", "grouped-head-matrix-bf16-probability-v1",
    ] {
    let descriptor = @luna_fused_artifact_admission.DecodeExecutionDescriptor::from_numeric_law(
      law,
      key_value_tile_rows=64,
      pipeline_stages=2,
    )
    let runtime = qwen_fused_test_runtime(
      plan,
      blueprint,
      phase_specific_attention=true,
      decode_execution=descriptor,
    )
    let spans = admit_reusable_qwen_fused_attention_spans(
      plan,
      @worker_protocol.ModelPlanGeneration::first(),
      blueprint,
      runtime,
    )
    assert_eq(spans.length(), 28)
    for span in spans {
      assert_eq(span.decode_function_symbol(), descriptor.function_symbol())
      assert_eq(
        span.decode_launch().block_x(),
        if law.contains("matrix") {
          32
        } else {
          64
        },
      )
      assert_eq(
        span.decode_launch().shared_memory_bytes(),
        if law.contains("matrix") {
          69648
        } else {
          66064
        },
      )
    }
  }
}
