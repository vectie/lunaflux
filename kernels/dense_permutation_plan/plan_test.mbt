///|
test "video patch order matches independently enumerated channel-major input" {
  let plan = @dense_permutation_plan.Permutation::new(
    [1, 2, 1, 1, 1, 2, 2, 2],
    [0, 2, 4, 6, 1, 3, 5, 7],
    max_elements=16L,
  )
  let expected = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]
  let inverse = plan.inverse()
  for i, input in expected {
    assert_eq(plan.source_index(i.to_int64()), input.to_int64())
    assert_eq(inverse.source_index(input.to_int64()), i.to_int64())
  }
  assert_false(plan.is_identity())
}

///|
test "audio streams remain separate across row-to-channel transpose" {
  let plan = @dense_permutation_plan.Permutation::new(
    [2, 3, 2],
    [0, 2, 1],
    max_elements=12L,
  )
  // Each stream has three [left-latent,right-latent] rows, not interleaved streams.
  let expected = [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11]
  for i, input in expected {
    assert_eq(plan.source_index(i.to_int64()), input.to_int64())
  }
}

///|
test "temporal and spatial patches preserve multiple batch and channel axes" {
  let pack = @dense_permutation_plan.Permutation::new(
    [2, 3, 2, 2, 2, 2, 2, 2],
    [0, 2, 4, 6, 1, 3, 5, 7],
    max_elements=384L,
  )
  let unpack = pack.inverse()
  // Independent oracle starts in unsplit [B,C,T,H,W] coordinates.
  for input in 0..<384 {
    let w = input % 4
    let h = input / 4 % 4
    let t = input / 16 % 4
    let c = input / 64 % 3
    let batch = input / 192
    let row = ((batch * 2 + t / 2) * 2 + h / 2) * 2 + w / 2
    let column = ((c * 2 + t % 2) * 2 + h % 2) * 2 + w % 2
    let output = (row * 24 + column).to_int64()
    assert_eq(pack.source_index(output), input.to_int64())
    assert_eq(unpack.source_index(input.to_int64()), output)
  }
}

///|
test "all rank-three permutations are bijective and inversion recovers every cell" {
  for shape in [[2, 3, 4], [1, 3, 4], [2, 1, 4], [2, 3, 1], [1, 1, 1]] {
    for
      order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
      ] {
      let plan = @dense_permutation_plan.Permutation::new(
        shape,
        order,
        max_elements=24L,
      )
      let seen = Array::make(plan.element_count().to_int(), false)
      let inverse = plan.inverse()
      for i in 0..<seen.length() {
        let input = plan.source_index(i.to_int64())
        assert_false(seen[input.to_int()])
        seen[input.to_int()] = true
        assert_eq(inverse.source_index(input), i.to_int64())
      }
      assert_true(seen.all(x => x))
    }
  }
}

///|
test "unit axes and contiguous axes collapse without changing source identity" {
  let shape = [2, 1, 3, 4]
  let order = [0, 2, 1, 3]
  let identity = @dense_permutation_plan.Permutation::new(
    shape,
    order,
    max_elements=24L,
  )
  shape[0] = 99
  order[0] = 3
  assert_true(identity.is_identity())
  assert_eq(identity.terms().length(), 1)
  assert_eq(identity.terms()[0].extent, 24L)
  assert_eq(identity.source_index(23L), 23L)
  assert_true(identity.inverse().is_identity())
  let partial = @dense_permutation_plan.Permutation::new(
    [2, 3, 4, 5],
    [2, 3, 0, 1],
    max_elements=120L,
  )
  assert_eq(partial.terms().length(), 2)
  assert_eq(partial.terms()[0].extent, 20L)
  assert_eq(partial.terms()[1].extent, 6L)
}

///|
test "malformed axes and overflowing extents reject before tensor allocation" {
  for
    (shape, order, limit) in [
      ([], [], 1L),
      ([0], [0], 1L),
      ([-1], [0], 1L),
      ([2, 3], [0, 0], 6L),
      ([2, 3], [0], 6L),
      ([2, 3], [0, 2], 6L),
      ([2, 3], [-1, 0], 6L),
      ([2, 3], [0, 1], 5L),
      ([1], [0], 0L),
      ([2147483647, 2147483647, 3], [0, 1, 2], 9223372036854775807L),
    ] {
    try
      @dense_permutation_plan.Permutation::new(shape, order, max_elements=limit)
    catch {
      @dense_permutation_plan.PermutationError::InvalidShape
      | @dense_permutation_plan.PermutationError::InvalidOrder
      | @dense_permutation_plan.PermutationError::ElementLimitExceeded => ()
      _ => fail("unexpected permutation error")
    } noraise {
      _ => fail("invalid permutation admitted")
    }
  }
  let plan = @dense_permutation_plan.Permutation::new(
    [2, 3],
    [1, 0],
    max_elements=6L,
  )
  for index in [-1L, 6L] {
    try plan.source_index(index) catch {
      @dense_permutation_plan.PermutationError::IndexOutOfBounds => ()
      _ => fail("wrong bounds error")
    } noraise {
      _ => fail("invalid index admitted")
    }
  }
}
