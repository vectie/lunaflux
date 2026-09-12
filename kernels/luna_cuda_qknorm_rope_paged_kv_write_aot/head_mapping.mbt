///|
const POSTPROCESS_HEADS_PER_WORKGROUP : Int = 4

///|
/// The production ABI admits the original one-head launch and the grouped
/// lowering. Keeping this law beside the producer prevents runtime admission
/// from freezing one backend layout into the execution graph.
pub fn supports_production_head_group_count(
  packed_heads : Int,
  group_count : Int,
) -> Bool {
  packed_heads > 0 &&
  (
    group_count == packed_heads ||
    group_count == (packed_heads - 1) / POSTPROCESS_HEADS_PER_WORKGROUP + 1
  )
}

///|
test "production head group admission covers grouped tails and pinned legacy launches" {
  for heads in [1, 3, 4, 5, 24, 32, 0x7fffffff] {
    assert_true(supports_production_head_group_count(heads, heads))
    assert_true(
      supports_production_head_group_count(
        heads,
        (heads - 1) / POSTPROCESS_HEADS_PER_WORKGROUP + 1,
      ),
    )
  }
  for groups in [0, 1, 7, 9, 16, 31, 33] {
    assert_false(supports_production_head_group_count(32, groups))
  }
  assert_false(supports_production_head_group_count(0, 0))
  assert_false(supports_production_head_group_count(-1, -1))
}
