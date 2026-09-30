///|
test "dual score ownership has a distinct explicit numerical law" {
  let fold = @attention_physical_ir.BlockwiseFold::new(
    dimensions=128,
    keys=32,
    score_owners=2,
  ).unwrap()
  assert_eq(fold.score_owners(), 2)
  assert_true(fold.rotates_score_groups())
  assert_eq(fold.numeric_law(), "dual-score-blockwise-f32-probability-v2")
  assert_eq(fold.dimensions(), (128, 32))
  assert_true(
    @attention_physical_ir.BlockwiseFold::new(
      dimensions=128,
      keys=32,
      score_owners=3,
    )
    is None,
  )
  let old = @attention_physical_ir.BlockwiseFold::new(dimensions=128, keys=32).unwrap()
  assert_eq(old.score_owners(), 1)
  assert_false(old.rotates_score_groups())
  assert_false(fold == old)
}

///|
test "partition merge shares statistics without changing ascending sum law" {
  let plan = @attention_physical_ir.PartitionMerge::new(
    partitions=8,
    dimensions=128,
  ).unwrap()
  assert_eq(plan.dimensions(), (8, 128))
  assert_eq(plan.steps(), [
    ReducePartitionMaximum,
    FormPartitionScalesAndDenominator,
    PublishPartitionStatistics,
    AccumulatePartitionNumerators,
    NormalizePartitionNumerators,
  ])
  assert_eq(plan.numeric_law(), "ascending-partition-f32-statistics-v1")
  assert_true(
    @attention_physical_ir.PartitionMerge::new(partitions=1, dimensions=128)
    is None,
  )
  assert_true(
    @attention_physical_ir.PartitionMerge::new(partitions=8, dimensions=0)
    is None,
  )
}

///|
/// Independent deterministic interpreter of the two disjoint score owners.
/// Each owner follows ascending component groups and a 16-owner tree.
fn dual_score_dot(
  q : ArrayView[Float],
  k : ArrayView[Float],
  owner : Int,
) -> Float {
  let partials = Array::make(16, 0.0F)
  let groups = (q.length() + 15) / 16
  for lane in 0..<16 {
    for item in 0..<groups {
      let component = lane + (item + owner) % groups * 16
      if component < q.length() {
        partials[lane] += q[component] * k[component]
      }
    }
  }
  for stride = 8; stride > 0; stride = stride / 2 {
    for lane in 0..<stride {
      partials[lane] += partials[lane + stride]
    }
  }
  partials[0]
}

///|
test "dual score reduction declares bounded error rather than bit equality" {
  for dimensions in [1, 17, 64, 80, 128, 160, 256] {
    let q = Array::makei(dimensions, i => {
      Float::from_int(i * 17 % 59 - 29) / 31.0F
    })
    for key in 0..<65 {
      let k = Array::makei(dimensions, i => {
        Float::from_int((i + key) * 31 % 67 - 33) / 37.0F
      })
      let expected = for component in 0..<dimensions; sum = 0.0 {
        continue sum + q[component].to_double() * k[component].to_double()
      } nobreak {
        sum
      }
      for owner in 0..<2 {
        let actual = dual_score_dot(q, k, owner)
        assert_true((actual.to_double() - expected).abs() <= 0.00001)
        assert_eq(actual, dual_score_dot(q, k, owner))
      }
    }
  }
}
