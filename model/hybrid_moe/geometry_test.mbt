///|
fn geometry() -> @hybrid_moe.Geometry raise {
  @hybrid_moe.Geometry::new(
    hidden_width=2048,
    vocabulary_size=248144,
    context_length=262144,
    layer_count=40,
    full_interval=4,
    full={ query_heads: 16, kv_heads: 2, head_width: 256, output_gate: true, },
    recurrent={
      key_heads: 16,
      key_width: 128,
      value_heads: 32,
      value_width: 128,
      convolution_width: 4,
    },
    experts={
      routed: 256,
      selected: 8,
      intermediate_width: 512,
      shared_width: 512,
    },
  )
}

///|
test "hybrid geometry keeps state and KV ownership distinct" {
  let g = geometry()
  assert_eq(g.full_layer_count(), 10)
  assert_eq(g.recurrent_layer_count(), 30)
  assert_true(g.layer_kind(2) is Some(GatedDeltaLinearAttention))
  assert_true(g.layer_kind(3) is Some(FullAttention))
  assert_true(g.layer_kind(40) is None)
  assert_eq(g.bf16_kv_bytes(1, 4096), 83886080L)
  assert_eq(g.fp32_recurrent_matrix_bytes(1), 62914560L)
  assert_eq(g.experts().routed, 256)
  assert_eq(g.experts().selected, 8)
}

///|
test "invalid cache capacities never wrap into smaller allocations" {
  let g = geometry()
  assert_true(
    try g.bf16_kv_bytes(1, 262145) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try g.fp32_recurrent_matrix_bytes(0) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
