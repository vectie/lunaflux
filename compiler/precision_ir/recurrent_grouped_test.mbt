///|
test "grouped delta describes scalar decay without clipping a model domain" {
  let plan = @precision_ir.RecurrentDeltaPrecision::new(
    head_count=32,
    query_key_heads=16,
    key_dimension=128,
    value_dimension=128,
    decay_layout=PerValueHead,
    maximum_rows=8,
    maximum_sequences=1,
    cache_slots=1,
    normalization_epsilon=0.000001,
    maximum_state_bytes=2097152L,
  )
  assert_eq(plan.query_key_head_count(), 16)
  assert_true(plan.decay_layout() is PerValueHead)
  assert_true(plan.log_decay_lower_bound().is_neg_inf())
  assert_eq(plan.state_bytes(), 2097152L)
  assert_true(
    try
      @precision_ir.RecurrentDeltaPrecision::new(
        head_count=3,
        query_key_heads=2,
        key_dimension=2,
        value_dimension=2,
        maximum_rows=1,
        maximum_sequences=1,
        cache_slots=1,
        normalization_epsilon=0.000001,
        maximum_state_bytes=1024L,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
