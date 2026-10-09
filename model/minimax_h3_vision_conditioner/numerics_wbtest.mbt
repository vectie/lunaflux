///|
test "learned position interpolation preserves each BF16 rounding boundary" {
  // Three half-ULP terms each tie back to the even 0.25 representation.
  // Accumulating in FP32 instead would incorrectly retain all three terms.
  let actual = interpolate_bf16(
    [1.0F, 0.00390625F, 0.00390625F, 0.00390625F][:],
    0.5F,
    0.5F,
  )
  assert_eq(actual, 0.25F)
  assert_eq(round_bf16(0.25F + 3.0F * 0.0009765625F), 0.25390625F)
  assert_eq(interpolate_bf16([1.0F, 2.0F, 3.0F, 4.0F][:], 0.0F, 0.0F), 1.0F)
  assert_eq(interpolate_bf16([1.0F, 2.0F, 3.0F, 4.0F][:], 1.0F, 1.0F), 4.0F)
  assert_eq(interpolate_bf16([1.0F, 2.0F, 3.0F, 4.0F][:], 0.5F, 0.5F), 2.5F)
}

///|
test "position stencil clamps endpoints and preserves processor merge order" {
  let p = VisionPlan::new(
    [{ time: 2, height: 4, width: 4, }],
    max_workspace_bytes=10000000L,
    max_output_bytes=10000000L,
  )
  let first = p.position_stencil(0)
  assert_eq(first[0], (0, 1.0F))
  // Row one is (y=0,x=1), not the next point in a row-major 4x4 table.
  let next = p.position_stencil(1)
  assert_eq(next[0].0, 15)
  assert_eq(next[1].0, 16)
  assert_true(next[0].1 > 0.333F && next[0].1 < 0.334F)
  let last = p.position_stencil(15)
  assert_eq(last[0], (2303, 1.0F))
  for item in last {
    assert_eq(item.0, 2303)
  }
  for row in 0..<16 {
    let a = p.position_stencil(row)
    let b = p.position_stencil(row + 16)
    let mut sum = 0.0F
    for i in 0..<4 {
      assert_eq(a[i], b[i])
      sum += a[i].1
    }
    assert_true(sum >= 0.999999F && sum <= 1.000001F)
  }
}
