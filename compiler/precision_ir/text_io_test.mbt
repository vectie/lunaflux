///|
test "selected text output rows do not scale logits with prefill length" {
  let plan = @precision_ir.TextIoPrecision::new(
    vocabulary_size=154880,
    hidden_width=4096,
    stream_count=4,
    maximum_rows=4096,
    maximum_output_rows=1,
    normalization_epsilon=1.0e-5,
    maximum_workspace_bytes=627712L,
  )
  assert_eq(plan.residual_bytes(), 134217728L)
  assert_eq(plan.normalized_bytes(), 8192L)
  assert_eq(plan.logits_bytes(), 619520L)
  assert_eq(plan.workspace_bytes(), 627712L)
  assert_eq(plan.table_bytes(), 1268776960L)
  assert_true(plan.normalization_law() is Bf16NormalizeThenWeight)
  assert_true(plan.logits_law() is Bf16Round)
  let failed = try {
    ignore(
      @precision_ir.TextIoPrecision::new(
        vocabulary_size=154880,
        hidden_width=4096,
        stream_count=4,
        maximum_rows=4096,
        maximum_output_rows=1,
        normalization_epsilon=1.0e-5,
        maximum_workspace_bytes=627711L,
      ),
    )
    false
  } catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => true
    _ => false
  }
  assert_true(failed)
}

///|
test "text ingress egress geometry rejects impossible output capacity" {
  let failed = try {
    ignore(
      @precision_ir.TextIoPrecision::new(
        vocabulary_size=11,
        hidden_width=3,
        stream_count=1,
        maximum_rows=2,
        maximum_output_rows=3,
        normalization_epsilon=1.0e-5,
        maximum_workspace_bytes=1024L,
      ),
    )
    false
  } catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => true
    _ => false
  }
  assert_true(failed)
}
