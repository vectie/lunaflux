///|
test "portable score ownership and contraction have separate numeric identities" {
  for owners in [2, 4, 8] {
    for contraction in [false, true] {
      let fold = @attention_physical_ir.BlockwiseFold::new(
        dimensions=128,
        keys=32,
        score_owners=owners,
        contraction~,
      ).unwrap()
      assert_eq(fold.score_owners(), owners)
      assert_eq(fold.contracts_multiply_add(), contraction)
      assert_true(fold.rotates_score_groups())
      if owners != 2 || contraction {
        assert_eq(
          fold.numeric_law(),
          "owned\{owners}-blockwise-\{if contraction { "fma-" } else { "" }}f32-probability-v4",
        )
      }
    }
  }
  for owners in [0, 3, 16, 33] {
    assert_true(
      @attention_physical_ir.BlockwiseFold::new(
        dimensions=128,
        keys=32,
        score_owners=owners,
      )
      is None,
    )
  }
}

///|
test "owned score cyclic component traversals are complete disjoint bijections" {
  for dimensions in [1, 17, 64, 80, 128, 160, 256] {
    for owners in [2, 4, 8] {
      let lanes = 32 / owners
      let groups = (dimensions + lanes - 1) / lanes
      for owner in 0..<owners {
        let seen = Array::make(dimensions, 0)
        for lane in 0..<lanes {
          for item in 0..<groups {
            let component = lane + (item + owner) % groups * lanes
            if component < dimensions {
              seen[component] += 1
            }
          }
        }
        assert_true(seen.all(count => count == 1))
      }
    }
  }
}
