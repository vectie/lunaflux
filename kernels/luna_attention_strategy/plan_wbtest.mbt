///|
test "attention planning separates query-tiled prefill and split decode" {
  let shape = AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let plan = plan_attention_dispatch(shape, 8192, 256, 8)
  assert_eq(plan.prefill().phase(), Prefill)
  assert_eq(plan.prefill().traversal(), QueryTile)
  assert_eq(plan.prefill().query_tile(), 16)
  assert_eq(plan.decode().phase(), Decode)
  assert_eq(plan.decode().traversal(), KeyValueSplit)
  assert_eq(plan.decode().reduction(), SplitOnlineSoftmaxMerge)
  assert_eq(plan.decode().key_value_splits(), 8)
}

///|
test "one-split decode workspace is the full online-softmax interval" {
  let shape = AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let dispatch = plan_attention_dispatch(shape, 1024, 32, 8)
  assert_eq(dispatch.decode().reduction(), OnlineSoftmax)
  let workspace = plan_decode_attention_workspace(
    shape,
    dispatch.decode(),
    4,
    1024,
    1048576L,
  )
  assert_eq(workspace.split_count(), 1)
  assert_eq(workspace.partitions().length(), 1)
  assert_eq(workspace.partitions()[0].split_index(), 0)
  assert_eq(workspace.partitions()[0].token_range(), (0, 1024))
  assert_eq(workspace.partial_count(), 64L)
  assert_eq(workspace.region_offsets_bytes(), (0L, 256L, 512L))
  assert_eq(workspace.total_bytes(), 33280L)
}

///|
test "multi-split decode partitions and merge order are deterministic" {
  let shape = AttentionShape::new(
    query_heads=8,
    kv_heads=2,
    head_dimension=64,
    tokens_per_page=16,
  )
  let dispatch = plan_attention_dispatch(shape, 4097, 32, 4)
  let workspace = plan_decode_attention_workspace(
    shape,
    dispatch.decode(),
    3,
    4097,
    1048576L,
  )
  assert_eq(workspace.split_count(), 4)
  assert_eq(workspace.partitions().length(), 4)
  assert_eq(workspace.partitions()[0].token_range(), (0, 1040))
  assert_eq(workspace.partitions()[1].token_range(), (1040, 2064))
  assert_eq(workspace.partitions()[2].token_range(), (2064, 3088))
  assert_eq(workspace.partitions()[3].token_range(), (3088, 4097))
  for index, partition in workspace.partitions() {
    assert_eq(partition.split_index(), index)
  }
}

///|
test "decode workspace rejects insufficient bytes and empty split geometry" {
  let shape = AttentionShape::new(
    query_heads=8,
    kv_heads=2,
    head_dimension=64,
    tokens_per_page=16,
  )
  let dispatch = plan_attention_dispatch(shape, 4096, 32, 4)
  try
    plan_decode_attention_workspace(shape, dispatch.decode(), 8, 4096, 16L)
  catch {
    InvalidAttentionStrategy(Workspace) => ()
    _ => fail("unexpected workspace rejection")
  } noraise {
    _ => fail("expected workspace rejection")
  }
  let tiny_dispatch = plan_attention_dispatch(shape, 1, 1, 1)
  try
    plan_decode_attention_workspace(shape, tiny_dispatch.prefill(), 1, 1, 4096L)
  catch {
    InvalidAttentionStrategy(Workspace) => ()
    _ => fail("unexpected phase rejection")
  } noraise {
    _ => fail("expected phase rejection")
  }
}
