///|
test "compact operand tiling is a bijection and preserves complete microtiles" {
  for rows in [16, 32, 64, 128] {
    for columns in [16, 32, 64, 128, 256] {
      let layout = @luna_attention_tile_schedule.attention_operand_tile_layout(
        rows, columns, 16,
      ).unwrap()
      let seen = Array::make(rows * columns, false)
      for row in 0..<rows {
        for column in 0..<columns {
          let offset = layout.offset(row, column).unwrap()
          assert_false(seen[offset])
          seen[offset] = true
          let base = layout.offset(row / 16 * 16, column / 16 * 16).unwrap()
          assert_eq(offset, base + row % 16 * 16 + column % 16)
          if column % 8 == 0 {
            assert_eq(offset % 8, 0)
            for element in 0..<8 {
              assert_eq(
                layout.offset(row, column + element),
                Some(offset + element),
              )
            }
          }
        }
      }
      assert_true(seen.iter().all(value => value))
      assert_eq(layout.offset(-1, 0), None)
      assert_eq(layout.offset(rows, 0), None)
      assert_eq(layout.offset(0, columns), None)
    }
  }
}

///|
test "operand tiling rejects partial or overflowing domains" {
  for
    geometry in [
      (0, 16, 16),
      (16, 0, 16),
      (16, 16, 0),
      (17, 32, 16),
      (32, 17, 16),
      (65536, 65536, 16),
    ] {
    let (rows, columns, side) = geometry
    assert_eq(
      @luna_attention_tile_schedule.attention_operand_tile_layout(
        rows, columns, side,
      ),
      None,
    )
  }
}

///|
test "compact packing preserves logical ordered QK and PV folds" {
  let queries = 32
  let keys = 64
  let dimension = 128
  let q_layout = @luna_attention_tile_schedule.attention_operand_tile_layout(
    queries, dimension, 16,
  ).unwrap()
  let kv_layout = @luna_attention_tile_schedule.attention_operand_tile_layout(
    keys, dimension, 16,
  ).unwrap()
  let p_layout = @luna_attention_tile_schedule.attention_operand_tile_layout(
    queries, keys, 16,
  ).unwrap()
  let q = Array::makei(queries * dimension, i => (i % 19 - 9).to_double() / 8.0)
  let k = Array::makei(keys * dimension, i => (i % 23 - 11).to_double() / 16.0)
  let v = Array::makei(keys * dimension, i => (i % 13 - 6).to_double() / 8.0)
  let qp = Array::make(q.length(), 0.0)
  let kp = Array::make(k.length(), 0.0)
  let vp = Array::make(v.length(), 0.0)
  for row in 0..<queries {
    for column in 0..<dimension {
      qp[q_layout.offset(row, column).unwrap()] = q[row * dimension + column]
    }
  }
  for row in 0..<keys {
    for column in 0..<dimension {
      let packed = kv_layout.offset(row, column).unwrap()
      kp[packed] = k[row * dimension + column]
      vp[packed] = v[row * dimension + column]
    }
  }
  let p = Array::make(queries * keys, 0.0)
  let pp = Array::make(p.length(), 0.0)
  for query in 0..<queries {
    for key in 0..<keys {
      let mut ordinary = 0.0
      let mut compact = 0.0
      for reduction in 0..<dimension {
        ordinary += q[query * dimension + reduction] *
          k[key * dimension + reduction]
        compact += qp[q_layout.offset(query, reduction).unwrap()] *
          kp[kv_layout.offset(key, reduction).unwrap()]
      }
      assert_eq(ordinary, compact)
      p[query * keys + key] = ordinary
      pp[p_layout.offset(query, key).unwrap()] = compact
    }
    for column in 0..<dimension {
      let mut ordinary = 0.0
      let mut compact = 0.0
      for reduction in 0..<keys {
        ordinary += p[query * keys + reduction] *
          v[reduction * dimension + column]
        compact += pp[p_layout.offset(query, reduction).unwrap()] *
          vp[kv_layout.offset(reduction, column).unwrap()]
      }
      assert_eq(ordinary, compact)
    }
  }
}
