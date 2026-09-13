///|
test "transport layout also separates every matrix-load subtransaction" {
  for columns in [16, 32, 64, 128] {
    let layout : CudaProjectionTransportLayout = { columns, }
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
  }
}

///|
test "sibling copy uses contiguous logical vectors and separate zero stores" {
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
  assert_true(
    source.contains("const int element = (threadIdx.x + slot * 512) * 8;"),
  )
  assert_false(source.contains("const int bytes = valid ? 16 : 0;"))
  assert_true(source.contains("st.shared.v4.u32"))
  assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
}

///|
test "register-plane handoff covers every result with disjoint banks" {
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

///|
test "operand transfer separates narrow reads and zero values from bypass copies" {
  assert_false(
    projection_operand_transfer_source(64, asynchronous=false).contains(
      "cp.async.cg",
    ),
  )
  for width in [16, 32, 64, 128] {
    let source = projection_operand_transfer_source(width)
    assert_true(source.contains("st.shared.v4.u32 [%0], {0,0,0,0}"))
    assert_false(source.contains("16, %2"))
    assert_eq(source.contains("ld.global.v4.u32"), width < 64)
    assert_eq(source.contains("cp.async.cg.shared.global"), width >= 64)
    let layout : CudaProjectionTransportLayout = { columns: width, }
    // A predicate can remove lanes, but cannot introduce a bank alias into
    // an already disjoint 128-byte store subtransaction. Exhaust row tails.
    for live in 0..<=16 {
      for first in 0..<(16 * width / 64) {
        let live_banks = FixedArray::make(32, false)
        let zero_banks = FixedArray::make(32, false)
        for lane in 0..<8 {
          let element = (first * 8 + lane) * 8
          let row = element / width
          let banks = if row < live { live_banks } else { zero_banks }
          for word in 0..<4 {
            let bank = (layout.offset(row, element % width) / 2 + word) % 32
            assert_false(banks[bank])
            banks[bank] = true
          }
        }
      }
    }
  }
}
