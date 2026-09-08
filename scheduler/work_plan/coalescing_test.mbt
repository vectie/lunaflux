///|
/// Enumerate the semantic KV-write trace, not model arithmetic. Every chunk
/// continues from its predecessor and only the terminal one demands output.
fn prefill_trace(
  computed : Int,
  target : Int,
  budget : Int,
  chunk : Int,
) -> (Array[Int], Array[Int], Int) raise {
  let writes : Array[Int] = []
  let outputs : Array[Int] = []
  for cursor = computed, steps = 0; cursor < target; {
    let count = @work_plan.prefill_query_count(
      computed=cursor,
      target~,
      token_budget=budget,
      chunk_limit=chunk,
      kv_token_capacity=target,
    )
    assert_true(count > 0 && count <= budget && count <= chunk)
    let work = @work_plan.RowWork::new(
      packed_start=0,
      computed=cursor,
      query_count=count,
      kv_token_capacity=target,
      output_demand=cursor + count == target,
    )
    for position in work.kv_write_start()..<work.kv_write_end() {
      writes.push(position)
    }
    if work.output_token_row() >= 0 {
      outputs.push(cursor + work.output_token_row())
    }
    continue work.context_end(), steps + 1
  } nobreak {
    (writes, outputs, steps)
  }
}

///|
test "prefill coalescing preserves ordered KV effects and final output demand" {
  for start in [0, 7, 256, 1023] {
    for target in [1024, 1528, 2049] {
      let (expected_writes, expected_outputs, _) = prefill_trace(
        start, target, 256, 256,
      )
      assert_eq(expected_outputs, [target - 1])
      for size in [1, 31, 255, 256, 512, 1024] {
        let (writes, outputs, _) = prefill_trace(start, target, size, size)
        assert_eq(writes, expected_writes)
        assert_eq(outputs, expected_outputs)
      }
    }
  }
  let (_, _, small_steps) = prefill_trace(0, 1528, 256, 256)
  let (_, _, medium_steps) = prefill_trace(0, 1528, 512, 512)
  let (_, _, large_steps) = prefill_trace(0, 1528, 1024, 1024)
  assert_eq((small_steps, medium_steps, large_steps), (6, 3, 2))
}

///|
test "larger chunk remains bounded by decode reservation and KV capacity" {
  for budget in [1, 31, 255, 256, 511, 512, 1016, 1024] {
    let count = @work_plan.prefill_query_count(
      computed=1000,
      target=2048,
      token_budget=budget,
      chunk_limit=1024,
      kv_token_capacity=1528,
    )
    assert_eq(count, Int::min(budget, 528))
  }
  assert_eq(
    @work_plan.prefill_query_count(
      computed=2147483600,
      target=2147483647,
      token_budget=1024,
      chunk_limit=1024,
      kv_token_capacity=2147483647,
    ),
    47,
  )
}
