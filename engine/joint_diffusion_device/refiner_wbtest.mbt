///|
test "refiner uses ungated pre-norm ordering with no modulation stage" {
  let stages = describe_refiner(0, [1, 2, 3, 4], 5, 6, [7, 8], 9).map(pair => {
    pair.1
  })
  assert_eq(stages.length(), 10)
  for i, stage in stages {
    assert_eq(i, stage)
  }
}

///|
test "named refiner description flattens to the identical production order" {
  let described = describe_refiner(0, [1, 2, 3, 4], 5, 6, [7, 8], 9)
  assert_eq(described.map(pair => pair.1), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  assert_true(described[0].0 is Norm1)
  assert_true(described[5].0 is AttentionResidual)
  assert_true(described[7].0 is GateUp)
  assert_true(described[9].0 is MlpResidual)
}

///|
test "unrotated QK norm omits rotary operands instead of binding dummy tables" {
  let plain = normalization_arguments(0, 1, 2, 3, [])
  assert_eq(plain.length(), 6)
  assert_eq(plain[4], 0)
  assert_eq(plain[5], 1)
  let rotated = normalization_arguments(0, 1, 2, 3, [4, 5])
  assert_eq(rotated.length(), 8)
  assert_eq(rotated[4], 4)
  assert_eq(rotated[6], 0)
}

///|
test "refiner capture keeps attention inner width distinct from hidden state" {
  let p = BlockWorkspaceLayout::new(
    rows=3,
    hidden_width=5376,
    head_count=56,
    head_dimension=128,
    intermediate_width=14336,
    max_bytes=1048576L,
  )
  assert_eq(p.refiner_capture_region(Qkv), (p.scratch_offset, 129024L))
  assert_eq(p.refiner_capture_region(QkNorm), (p.scratch_offset, 129024L))
  assert_eq(
    p.refiner_capture_region(Attention),
    (p.scratch_offset + 129024L, 43008L),
  )
  assert_eq(p.refiner_capture_region(Projection).1, 32256L)
  assert_eq(p.refiner_capture_region(GateUp).1, 86016L)
}
