///|
test "independent head packing preserves complete numerical folds and tails" {
  let numerics = ProjectionAttentionIngressEpilogue::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    norm_epsilon=0.000001,
    rope_theta=1000000.0,
  ).numerics(reduction_lanes=32)
  for heads in [1, 3, 4, 5, 7, 8, 31, 32, 33, 127] {
    let packed = numerics.group_heads(heads~, workgroup_lanes=128)
    let (groups, heads_per_group, lanes) = packed.geometry()
    assert_eq(heads_per_group, 4)
    assert_eq(lanes, 32)
    let seen : Array[Int] = []
    for group in 0..<groups {
      for subgroup in 0..<heads_per_group {
        if packed.head_at(group, subgroup) is Some(head) {
          seen.push(head)
        }
      }
    }
    assert_eq(seen, Array::makei(heads, i => i))
    assert_eq(packed.head_at(-1, 0), None)
    assert_eq(packed.head_at(groups, 0), None)
    assert_eq(packed.head_at(0, -1), None)
    assert_eq(packed.head_at(0, heads_per_group), None)
  }
  let alternative = ProjectionAttentionIngressEpilogue::new(
      query_heads=8,
      kv_heads=4,
      head_dimension=64,
      norm_epsilon=0.000001,
      rope_theta=10000.0,
    )
    .numerics(reduction_lanes=16)
    .group_heads(heads=17, workgroup_lanes=64)
  assert_eq(alternative.geometry(), (5, 4, 16))
  assert_eq(alternative.head_at(4, 0), Some(16))
  assert_eq(alternative.head_at(4, 1), None)
  let maximum = numerics.group_heads(heads=2147483647, workgroup_lanes=128)
  assert_eq(maximum.head_at(536870911, 2), Some(2147483646))
  assert_eq(maximum.head_at(536870911, 3), None)
}

///|
test "head packing rejects incomplete reduction subgroups" {
  let numerics = ProjectionAttentionIngressEpilogue::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    norm_epsilon=0.000001,
    rope_theta=1000000.0,
  ).numerics(reduction_lanes=32)
  for pair in [(0, 128), (1, 0), (1, 16), (1, 33), (1, 2049)] {
    let (heads, workgroup_lanes) = pair
    try numerics.group_heads(heads~, workgroup_lanes~) catch {
      InvalidProjectionTileCompilation(InvalidProblem) => ()
      _ => fail("unexpected invalid head-map error")
    } noraise {
      _ => fail("invalid head map accepted")
    }
  }
}

///|
test "logical ingress ownership does not assume a CUDA block limit" {
  let epilogue = ProjectionAttentionIngressEpilogue::new(
    query_heads=1,
    kv_heads=1,
    head_dimension=128,
    norm_epsilon=0.000001,
    rope_theta=10000.0,
  )
  for lanes in [1, 32, 64, 1024, 2048, 1073741824] {
    let numerics = epilogue.numerics(reduction_lanes=lanes)
    assert_eq(numerics.shape(), (128, lanes))
    let map = numerics.group_heads(heads=2147483647, workgroup_lanes=lanes)
    assert_eq(map.geometry(), (2147483647, 1, lanes))
    assert_eq(map.head_at(2147483646, 0), Some(2147483646))
    assert_eq(map.head_at(2147483647, 0), None)
  }
  let map = epilogue
    .numerics(reduction_lanes=32)
    .group_heads(heads=129, workgroup_lanes=4096)
  assert_eq(map.geometry(), (2, 128, 32))
  assert_eq(map.head_at(1, 0), Some(128))
  assert_eq(map.head_at(1, 1), None)
}
