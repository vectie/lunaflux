///|
fn tile_ir_test_plan(
  causal : Bool,
) -> @luna_attention_strategy.CompiledAttentionTilePlan raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=128,
    context_tokens=4096,
    causal~,
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=262144L,
    max_pipeline_stages=3,
    supports_async_copy=true,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  )
  @luna_attention_strategy.compile_attention_tile_plan(
    [],
    backend=@luna_attention_strategy.AttentionBackendId::new("portable"),
    target=@luna_attention_strategy.AttentionTargetId::new("fixture"),
    problem~,
    capabilities~,
  )
}

///|
test "functional attention tile IR encodes online softmax as an explicit fold" {
  let first = compile_attention_tile_ir(tile_ir_test_plan(true))
  let repeated = compile_attention_tile_ir(tile_ir_test_plan(true))
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.canonical_bytes(), repeated.canonical_bytes())
  assert_eq(first.bindings().length(), 9)
  assert_eq(first.root().as_int(), 9)
  assert_eq(
    first.bindings()[3].expression(),
    AttentionTileDot(
      AttentionTileValueId::from_stable_int(1),
      AttentionTileValueId::from_stable_int(2),
    ),
  )
  assert_eq(
    first.bindings()[5].expression(),
    AttentionTileCausalMask(AttentionTileValueId::from_stable_int(5)),
  )
  assert_eq(
    first.bindings()[7].expression(),
    AttentionTileFoldKeyValueStream(AttentionTileValueId::from_stable_int(7)),
  )
  assert_eq(first.bindings()[8].kind(), OutputTileValue)
  assert_eq(first.digest().as_hex().length(), 64)
}

///|
test "noncausal tile IR removes the mask without changing online reduction" {
  let causal = compile_attention_tile_ir(tile_ir_test_plan(true))
  let noncausal = compile_attention_tile_ir(tile_ir_test_plan(false))
  assert_eq(noncausal.bindings().length(), 8)
  assert_eq(noncausal.root().as_int(), 8)
  assert_eq(
    noncausal.bindings()[5].expression(),
    AttentionTileOnlineSoftmax(
      scores=AttentionTileValueId::from_stable_int(5),
      values=AttentionTileValueId::from_stable_int(3),
    ),
  )
  assert_eq(
    noncausal.bindings()[6].expression(),
    AttentionTileFoldKeyValueStream(AttentionTileValueId::from_stable_int(6)),
  )
  assert_true(causal.digest() != noncausal.digest())
}

///|
test "attention tile value IDs reject unstable ranges" {
  try AttentionTileValueId::from_stable_int(0) catch {
    InvalidAttentionTileIr(InvalidValueId) => ()
    _ => fail("unexpected value-id rejection")
  } noraise {
    _ => fail("invalid value id was accepted")
  }
}
