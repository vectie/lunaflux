///|
test "FP4 bulk producer and fragment consumer agree across XOR vectors" {
  let layout = @physical_tile_ir.OperandLayout::new(
    columns=128,
    vector=16,
    rotations=8,
    row_group=1,
  ).unwrap()
  let storage = @physical_tile_ir.OperandStorage::new(
    layout,
    [16, 32],
    owners=128,
    stages=1,
  ).unwrap()
  assert_eq(storage.allocation_elements(), 6144)
  for operand in [0, 1] {
    let region = storage.region(operand).unwrap()
    let (base, rows, _) = region.geometry()
    for row in 0..<rows {
      for kk in [0, 64, 128, 192] {
        for t in 0..<4 {
          for half in [0, 16] {
            let logical = kk / 2 + t * 4 + half
            let consumer = base + row * 128 + (logical ^ (row % 8 * 16))
            assert_eq(region.offset(row, logical), Some(consumer))
          }
        }
      }
    }
  }
}

///|
test "FP4 staging tails keep unaligned scale rows out of async copies" {
  for inner in [64, 128, 192, 256, 320, 384, 1024, 4096] {
    let source = fp4_fragment_pipeline(
      0, inner, 128, 2, 65536L, 131072L, "probe",
    )
    assert_true(source.contains("cp.async.wait_group 0"))
    assert_true(source.contains("base_k+kk<\{inner}"))
    assert_true(source.contains("bytes?offset:0"))
    assert_eq(source.contains("valid&&offset<"), inner % 256 != 0)
    assert_true(source.contains("__syncthreads();\n}\n}"))
  }
}

///|
test "paired FP4 owners stage left once without joining reduction chains" {
  for inner in [64, 128, 192, 256, 320, 384, 1024, 4096] {
    let source = fp4_fragment_pipeline(
      0,
      inner,
      1024,
      8,
      65536L,
      131072L,
      "paired",
      work_tile="work_tile",
      paired_weight_scales=262144L,
    )
    assert_true(source.contains("unsigned char tile[11520]"))
    assert_eq(source.split("for(int base_k=").count(), 2)
    assert_eq(source.split("cp.async.wait_group 0").count(), 2)
    assert_eq(source.split("paired_mma(d0").count(), 2)
    assert_eq(source.split("paired_mma(d1").count(), 2)
    assert_true(
      source.contains("w0+131072ULL") || source.contains("w0[131072ULL"),
    )
    assert_true(
      source.contains("w1+262144ULL") || source.contains("w1[262144ULL"),
    )
    assert_true(source.contains("tile+11008+wr*16+kk/16"))
    assert_true(source.contains("__syncthreads();\n}\n}"))
  }
}

///|
test "wider column ownership covers every logical cell once including output tails" {
  for columns in [32, 64, 128] {
    for extent in [1, 31, 32, 33, 63, 65, 127, 128, 129] {
      let visits = Array::make(16 * extent, 0)
      for tile in 0..<((extent + columns - 1) / columns) {
        for thread in 0..<128 {
          let lane = thread % 32
          for i in 0..<(columns / 8) {
            let row = lane / 4 + i % 4 / 2 * 8
            let col = tile * columns +
              thread / 32 * 8 +
              i / 4 * 32 +
              lane % 4 * 2 +
              i % 2
            if col < extent {
              visits[row * extent + col] += 1
            }
          }
        }
      }
      assert_true(visits.all(n => n == 1))
    }
    let source = fp4_fragment_pipeline(
      2,
      1024,
      4096,
      8,
      65536L,
      131072L,
      "wide",
      tile_columns=columns,
    )
    if columns == 32 {
      assert_false(source.contains("fragment<"))
      assert_true(source.contains("wide_mma(d2,a0p[lo]"))
    } else {
      assert_true(source.contains("fragment<\{columns/32}"))
      assert_true(source.contains("wide_mma(d2+fragment*4"))
    }
    assert_true(source.contains("__syncthreads();\n}\n}"))
  }
}
