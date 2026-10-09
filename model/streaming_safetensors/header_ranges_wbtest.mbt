///|
fn range_test_tensor(
  name : String,
  start : Int64,
  end : Int64,
) -> StreamingSafetensorsTensorMetadata {
  {
    name,
    dtype: StreamingSafetensorsDTypeSpec::float32(),
    shape: ReadOnlyArray::from_array([]),
    data_start: start,
    data_end: end,
  }
}

///|
test "sorted range validation preserves header order and allows adjacent or gapped ranges" {
  let tensors = [
    range_test_tensor("last", 24L, 28L),
    range_test_tensor("first", 0L, 4L),
    range_test_tensor("middle", 4L, 8L),
  ]
  validate_tensor_ranges(tensors, 3)
  assert_eq(tensors.map(tensor => tensor.name), ["last", "first", "middle"])
  validate_tensor_ranges([], 0)
}

///|
test "range validation reports original ordinals for nested equal-start and reversed overlaps" {
  for
    (left_start, left_end, right_start, right_end) in [
      (8L, 24L, 12L, 16L),
      (8L, 12L, 8L, 16L),
      (12L, 16L, 8L, 24L),
    ] {
    try
      validate_tensor_ranges(
        [
          range_test_tensor("a", left_start, left_end),
          range_test_tensor("b", right_start, right_end),
        ],
        5,
      )
    catch {
      OverlappingTensorRanges(shard=5, first=0, second=1) => ()
      _ => fail("incorrect overlap coordinates")
    } noraise {
      _ => fail("overlap accepted")
    }
  }
}

///|
test "range validation agrees with pairwise oracle across deterministic permutations" {
  for seed in 0..<64 {
    let tensors = Array::makei(23, ordinal => {
      let position = (ordinal * 7 + seed) % 23
      let start = position.to_int64() * 8L
      let width = if seed % 2 == 0 { 4L } else { 12L }
      range_test_tensor("tensor_\{ordinal}", start, start + width)
    })
    let overlaps = tensors.any(a => {
      tensors.any(b => {
        a.name != b.name &&
        a.data_start < b.data_end &&
        b.data_start < a.data_end
      })
    })
    try validate_tensor_ranges(tensors, 0) catch {
      OverlappingTensorRanges(shard=0, first~, second~) => {
        assert_true(overlaps)
        assert_true(first < second)
        let a = tensors[first]
        let b = tensors[second]
        assert_true(a.data_start < b.data_end && b.data_start < a.data_end)
      }
      _ => fail("unexpected range error")
    } noraise {
      _ => assert_false(overlaps)
    }
  }
}
