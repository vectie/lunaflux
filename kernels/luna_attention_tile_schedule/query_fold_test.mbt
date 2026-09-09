///|
test "finite query fold product covers each active query exactly once" {
  for tile in [1, 16, 17, 32, 64] {
    for subgroups in [1, 4, 8, 16, 32] {
      let offsets = @luna_attention_tile_schedule.scalarize_attention_query_fold(
        tile, subgroups,
      )
      for active in 0..<=tile {
        let visits = Array::make(tile, 0)
        for offset in offsets {
          for subgroup in 0..<subgroups {
            let query = offset + subgroup
            if query < active {
              visits[query] += 1
            }
          }
        }
        for query in 0..<tile {
          assert_eq(visits[query], if query < active { 1 } else { 0 })
        }
      }
    }
  }
}

///|
test "finite query fold rejects unbounded or empty geometry" {
  for geometry in [(0, 8), (65, 8), (32, 0), (32, 33)] {
    let (tile, groups) = geometry
    try
      @luna_attention_tile_schedule.scalarize_attention_query_fold(tile, groups)
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid fold geometry accepted")
    }
  }
}

///|
test "finite product preserves each masked ordered state recurrence" {
  for active in [1, 7, 8, 9, 17, 31, 32] {
    let original = Array::make(32, 0.0)
    let product = Array::make(32, 0.0)
    let offsets = @luna_attention_tile_schedule.scalarize_attention_query_fold(
      32, 8,
    )
    // The state update deliberately depends on the previous value and key
    // order, so a reduction reassociation or duplicated tail changes equality.
    for key in 0..<73 {
      for subgroup in 0..<8 {
        for query = subgroup; query < active; query = query + 8 {
          original[query] = original[query] * 0.875 +
            (query + key * 3).to_double() / 17.0
        }
      }
      for offset in offsets {
        for subgroup in 0..<8 {
          let query = subgroup + offset
          if query < active {
            product[query] = product[query] * 0.875 +
              (query + key * 3).to_double() / 17.0
          }
        }
      }
    }
    assert_eq(original, product)
  }
}
