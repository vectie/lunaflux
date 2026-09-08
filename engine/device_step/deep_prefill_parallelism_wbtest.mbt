///|
test "deep prefill preserves sparse query frontier but rejects dense work" {
  assert_true(
    paged_partitioned_prefill_frontier_has_enough_work(
      1, 16, 65536, 1024, 16, 8,
    ),
  )
  assert_true(
    paged_partitioned_prefill_frontier_has_enough_work(
      8, 128, 8192, 1024, 16, 8,
    ),
  )
  assert_false(
    paged_partitioned_prefill_frontier_has_enough_work(
      8, 129, 65536, 1024, 16, 8,
    ),
  )
  assert_false(
    paged_partitioned_prefill_frontier_has_enough_work(
      9, 128, 65536, 1024, 16, 8,
    ),
  )
  for tokens in [256, 512, 1024, 4096] {
    assert_false(
      paged_partitioned_prefill_frontier_has_enough_work(
        8, tokens, 1048576, 1024, 16, 8,
      ),
    )
  }
}

///|
test "deep prefill density guard is independent of tile units" {
  for tile in [16, 32, 64, 128] {
    assert_true(
      paged_partitioned_prefill_frontier_has_enough_work(
        8,
        tile * 8,
        8192,
        1024,
        tile,
        8,
      ),
    )
    assert_false(
      paged_partitioned_prefill_frontier_has_enough_work(
        8,
        tile * 8 + 1,
        1048576,
        1024,
        tile,
        8,
      ),
    )
    assert_false(
      paged_partitioned_prefill_frontier_has_enough_work(
        8,
        tile * 8,
        4096,
        1024,
        tile,
        8,
      ),
    )
  }
}
