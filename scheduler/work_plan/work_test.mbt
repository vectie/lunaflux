///|
test "work ranges preserve KV effects when output is absent" {
  let work = RowWork::new(
    packed_start=17,
    computed=256,
    query_count=128,
    kv_token_capacity=512,
    output_demand=false,
  )
  assert_eq(
    (work.query_start(), work.query_end(), work.query_count()),
    (17, 145, 128),
  )
  assert_eq((work.cached_context_end(), work.context_end()), (256, 384))
  assert_eq((work.kv_write_start(), work.kv_write_end()), (256, 384))
  assert_eq(work.output_token_row(), -1)
  let terminal = RowWork::new(
    packed_start=17,
    computed=256,
    query_count=128,
    kv_token_capacity=512,
    output_demand=true,
  )
  assert_eq(terminal.output_token_row(), 144)
  assert_eq(terminal.kv_write_end(), work.kv_write_end())
}

///|
test "progress planner respects each independent bound" {
  for budget in [0, 1, 8, 256, 512] {
    for capacity in [0, 255, 256, 257, 512, 1024] {
      let count = prefill_query_count(
        computed=256,
        target=768,
        token_budget=budget,
        chunk_limit=256,
        kv_token_capacity=capacity,
      )
      assert_true(count >= 0 && count <= budget && count <= 256)
      if count > 0 {
        assert_true(256 + count <= capacity && 256 + count <= 768)
      }
    }
  }
  assert_eq(
    prefill_query_count(
      computed=256,
      target=768,
      token_budget=512,
      chunk_limit=256,
      kv_token_capacity=384,
    ),
    128,
  )
  assert_eq(
    prefill_query_count(
      computed=768,
      target=768,
      token_budget=512,
      chunk_limit=256,
      kv_token_capacity=1024,
    ),
    0,
  )
}

///|
test "work construction rejects overflow before arithmetic" {
  try
    RowWork::new(
      packed_start=2147483647,
      computed=0,
      query_count=1,
      kv_token_capacity=1,
      output_demand=true,
    )
  catch {
    InvalidRange => ()
  } noraise {
    _ => fail("packed offset overflow accepted")
  }
  try
    RowWork::new(
      packed_start=0,
      computed=2147483647,
      query_count=1,
      kv_token_capacity=2147483647,
      output_demand=true,
    )
  catch {
    InvalidRange => ()
  } noraise {
    _ => fail("KV offset overflow accepted")
  }
  let last = RowWork::new(
    packed_start=0,
    computed=2147483646,
    query_count=1,
    kv_token_capacity=2147483647,
    output_demand=true,
  )
  assert_eq(last.context_end(), 2147483647)
}
