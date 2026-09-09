///|
test "eight-wide sibling operands preserve vectors and every matrix fragment" {
  for columns in [16, 32, 64, 128] {
    let layout : CudaProjectionMicrotileLayout = { columns, }
    for rows in [16, 32, 64, 192] {
      let seen = FixedArray::make(rows * columns, false)
      for row in 0..<rows {
        for column in 0..<columns {
          let address = layout.offset(row, column)
          assert_true(address >= 0 && address < seen.length())
          assert_false(seen[address])
          seen[address] = true
          if column % 8 == 0 {
            assert_eq(address % 8, 0)
            for lane in 0..<8 {
              assert_eq(layout.offset(row, column + lane), address + lane)
            }
          }
          let tile = layout.offset(row / 8 * 8, column / 8 * 8)
          assert_eq(tile % 64, 0)
          assert_eq(address, tile + row % 8 * 8 + column % 8)
        }
      }
      for present in seen {
        assert_true(present)
      }
    }
  }
}

///|
test "sibling source shares compact copy and matrix addresses without arena growth" {
  let parameters : ProjectionParameters = {
    family: GatedMlp,
    input_width: 1024,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 3072,
  }
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let source = row_variant_primary_test_source(
    parameters,
    runtime.dispatch_plan(1024),
  )
  assert_true(source.contains("sibling_storage[12288]"))
  assert_true(source.contains("destination + operand_index(element / 32, k)"))
  assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
  assert_true(
    source.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"),
  )
  assert_true(source.contains("inner += 32"))
  assert_true(source.contains("k += 16"))
  assert_true(source.contains("float c[2][8]"))
}

///|
test "matrix load subtransactions and producer vectors have disjoint banks" {
  for columns in [16, 32, 64, 128] {
    let layout : CudaProjectionMicrotileLayout = { columns, }
    for tile_row in 0..<24 {
      for tile_column in 0..<(columns / 8) {
        let banks = FixedArray::make(32, false)
        for row in 0..<8 {
          for word in 0..<4 {
            let bank = layout.offset(
                tile_row * 8 + row,
                tile_column * 8 + word * 2,
              ) /
              2 %
              32
            assert_false(banks[bank])
            banks[bank] = true
          }
        }
      }
    }
    // Producer assignment is the inverse layout at aligned vector boundaries.
    for vector in 0..<(192 * columns / 8) {
      let row = vector / 8 / (columns / 8) * 8 + vector % 8
      let column = vector / 8 % (columns / 8) * 8
      assert_eq(layout.offset(row, column), vector * 8)
    }
  }
}

///|
test "register-plane handoff is bank-disjoint and covers each output exactly once" {
  let seen = FixedArray::make(256, false)
  for value in 0..<8 {
    let banks = FixedArray::make(32, false)
    for lane in 0..<32 {
      let bank = (value * 32 + lane) % 32
      assert_false(banks[bank])
      banks[bank] = true
      let row = lane / 4 + value % 4 / 2 * 8
      let column = lane % 4 * 2 + value % 2 + value / 4 * 8
      assert_false(seen[row * 16 + column])
      seen[row * 16 + column] = true
    }
  }
  for present in seen {
    assert_true(present)
  }
}
