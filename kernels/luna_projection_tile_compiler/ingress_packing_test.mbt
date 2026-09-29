///|
test "packed producer owns each segmented weight row exactly once including tails" {
  for dimension in [16, 32, 64, 128] {
    for heads_per_group in [1, 2, 3, 4, 8] {
      let spec = ProjectionAttentionIngressEpilogue::new(
        query_heads=6,
        kv_heads=3,
        head_dimension=dimension,
        norm_epsilon=0.000001,
        rope_theta=10000.0,
      )
      let plan = spec.packing(heads_per_group~, reduction_lanes=32)
      let (groups, heads, columns, pairs) = plan.geometry()
      assert_eq(heads, heads_per_group)
      assert_eq(columns, dimension * heads)
      assert_eq(pairs, (dimension / 2 - 1) / 32 + 1)
      let seen : Array[(Int, Int)] = []
      for group in 0..<groups {
        for column in 0..<columns {
          if plan.weight_row(group, column) is Some(row) {
            seen.push(row)
          }
        }
      }
      let expected : Array[(Int, Int)] = []
      for segment, count in [6, 3, 3] {
        for row in 0..<(count * dimension) {
          expected.push((segment, row))
        }
      }
      assert_eq(seen, expected)
      assert_eq(plan.weight_row(-1, 0), None)
      assert_eq(plan.weight_row(groups, 0), None)
      assert_eq(plan.weight_row(0, -1), None)
      assert_eq(plan.weight_row(0, columns), None)
      assert_eq(plan.effects(), [
        PrepareRowRotary,
        ConsumeCompleteHeads,
        RetireRowRotary,
      ])
      assert_eq(
        plan.canonical_bytes(),
        spec.packing(heads_per_group~, reduction_lanes=32).canonical_bytes(),
      )
    }
  }
}

///|
test "head packing rejects overflow and zero ownership before materialization" {
  let spec = ProjectionAttentionIngressEpilogue::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    norm_epsilon=0.000001,
    rope_theta=10000.0,
  )
  for heads_per_group in [0, -1, 16777216] {
    try spec.packing(heads_per_group~, reduction_lanes=32) catch {
      InvalidProjectionTileCompilation(InvalidProblem) => ()
      _ => fail("wrong ownership rejection")
    } noraise {
      _ => fail("invalid head packing accepted")
    }
  }
}
