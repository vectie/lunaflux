///|
test "sparse decoder composition accounts for output and residual ownership" {
  let attention = @precision_ir.IndexedSparseAttentionPrecision::new(
    index_head_count=2,
    index_head_dimension=4,
    query_head_count=2,
    key_value_head_count=2,
    query_key_dimension=4,
    value_dimension=4,
    pool_size=4,
    top_k=8,
    append_incomplete_tail=true,
    maximum_queries=8,
    maximum_history=32,
    maximum_workspace_bytes=100000L,
  )
  let projection = @precision_ir.IndexedSparseProjectionPrecision::new(
    attention,
    hidden_width=8,
    query_rank=4,
    key_value_rank=2,
    rms_epsilon=1.0e-5,
    index_layer_norm_epsilon=1.0e-5,
    maximum_workspace_bytes=1184L,
  )
  let envelope = @precision_ir.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=8,
    maximum_rows=8,
    sinkhorn_iterations=20,
    control_epsilon=1.0e-6,
    normalization_epsilon=1.0e-5,
    mix_law=RoundedBf16ControlsAndProducts,
  )
  let exact = projection.workspace_bytes() +
    attention.workspace_bytes() +
    envelope.workspace_bytes() +
    128L
  let plan = @precision_ir.IndexedSparseDecoderPrecision::new(
    projection,
    envelope,
    maximum_state_bytes=10000L,
    maximum_workspace_bytes=exact,
  )
  assert_eq(plan.workspace_bytes(), exact)
  assert_eq(plan.attention_output_bytes(), 128L)
  assert_eq(plan.output_weight_bytes(), 128L)
  assert_eq(plan.history().state_bytes(), 1728L)
  try
    @precision_ir.IndexedSparseDecoderPrecision::new(
      projection,
      envelope,
      maximum_state_bytes=10000L,
      maximum_workspace_bytes=exact - 1L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("intermediate budget ignored")
  }
  let wrong = @precision_ir.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=16,
    maximum_rows=8,
    sinkhorn_iterations=20,
    control_epsilon=1.0e-6,
    normalization_epsilon=1.0e-5,
    mix_law=RoundedBf16ControlsAndProducts,
  )
  try
    @precision_ir.IndexedSparseDecoderPrecision::new(
      projection,
      wrong,
      maximum_state_bytes=10000L,
      maximum_workspace_bytes=100000L,
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("residual width mismatch")
  }
}
