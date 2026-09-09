///|
test "compact sibling operands preserve vectors and every matrix fragment" {
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
          let tile = layout.offset(row / 16 * 16, column / 16 * 16)
          assert_eq(tile % 256, 0)
          assert_eq(address, tile + row % 16 * 16 + column % 16)
        }
      }
      for present in seen {
        assert_true(present)
      }
    }
  }
}

///|
test "sibling source shares compact copy and WMMA addresses without arena growth" {
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
  assert_true(source.contains("operand_index(consumer_column * 16, k), 16"))
  assert_true(
    source.contains("operand_index((consumer_row + row) * 16, k), 16"),
  )
  assert_true(source.contains("inner += 32"))
  assert_true(source.contains("k += 16"))
  assert_true(source.contains("float> c[2]"))
}
