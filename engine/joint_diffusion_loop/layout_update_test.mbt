///|
test "RF state stays packed for all steps and returns bit-exact raw layout" {
  let pack = @dense_permutation_plan.Permutation::new(
    [1, 2, 1, 1, 1, 2, 2, 2],
    [0, 2, 4, 6, 1, 3, 5, 7],
    max_elements=16L,
  )
  let unpack = pack.inverse()
  let raw = FixedArray::makei(16, i => Float::from_double(i.to_double() / 7.0))
  let packed = FixedArray::makei(16, i => {
    raw[(try! pack.source_index(i.to_int64())).to_int()]
  })
  for step in steps() {
    let coefficients = step.video_coefficients()
    // A deterministic pointwise prediction allows an independent raw-layout
    // execution. This verifies update/reorder composition, not a real denoiser.
    for i in 0..<16 {
      raw[i] = coefficients.update(raw[i], raw[i] * 0.001F)
      packed[i] = coefficients.update(packed[i], packed[i] * 0.001F)
    }
  }
  for i in 0..<16 {
    let restored = packed[unpack.source_index(i.to_int64()).to_int()]
    assert_eq(restored.reinterpret_as_uint(), raw[i].reinterpret_as_uint())
  }
}
