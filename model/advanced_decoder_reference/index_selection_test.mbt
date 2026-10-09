///|
test "learned index scoring applies ReLU head weighting and stable top-k" {
  let selected = learned_index_top_k(
    [1.0F, 0.0F, 0.0F, 1.0F],
    [1.0F, 1.0F, 2.0F, -1.0F, -1.0F, 3.0F],
    [1.0F, 2.0F],
    head_count=2,
    head_dimension=2,
    top_k=2,
    limits=test_limits(),
  )
  assert_eq(selected.indices(), [2, 0])
  assert_eq(selected.scores(), [6.0F, 3.0F])
  let ties = learned_index_top_k(
    [0.0F],
    [1.0F, 2.0F, 3.0F],
    [1.0F],
    head_count=1,
    head_dimension=1,
    top_k=2,
    limits=test_limits(),
  )
  assert_eq(ties.indices(), [0, 1])
}

///|
test "deterministic compression returns only causal complete slots" {
  let limits = test_limits()
  assert_eq(
    deterministic_compressed_indices(
      compression_ratio=4,
      query_position=0,
      cache_offset=10,
      limits~,
    ),
    [],
  )
  assert_eq(
    deterministic_compressed_indices(
      compression_ratio=4,
      query_position=3,
      cache_offset=10,
      limits~,
    ),
    [10],
  )
  assert_eq(
    deterministic_compressed_indices(
      compression_ratio=4,
      query_position=8,
      cache_offset=10,
      limits~,
    ),
    [10, 11],
  )
}

///|
test "index selection rejects hostile dimensions bounds and NaN" {
  try
    learned_index_top_k(
      [1.0F],
      [1.0F],
      [@float.not_a_number],
      head_count=1,
      head_dimension=1,
      top_k=1,
      limits=test_limits(),
    )
  catch {
    NonFiniteInput(0) => ()
    _ => fail("unexpected learned-index NaN error")
  } noraise {
    _ => fail("learned index admitted NaN weight")
  }
  try
    deterministic_compressed_indices(
      compression_ratio=1,
      query_position=8,
      cache_offset=0,
      limits=test_limits(),
    )
  catch {
    InvalidCompression => ()
    _ => fail("unexpected compression error")
  } noraise {
    _ => fail("ratio-one compression was admitted")
  }
  try
    deterministic_compressed_indices(
      compression_ratio=4,
      query_position=7,
      cache_offset=2147483647,
      limits=test_limits(),
    )
  catch {
    InvalidCompression => ()
    _ => fail("unexpected compressed-index overflow error")
  } noraise {
    _ => fail("overflowing compressed index was admitted")
  }
}
