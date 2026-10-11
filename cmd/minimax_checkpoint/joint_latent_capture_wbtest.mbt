///|
test "terminal latent capture preserves F32 bits across canonical layout inversion" {
  for
    (shape, order) in [
      ([2, 3, 2], [0, 2, 1]),
      ([1, 2, 1, 1, 1, 2, 2, 2], [0, 2, 4, 6, 1, 3, 5, 7]),
    ] {
    let pack = @permutation.Permutation::new(shape, order, max_elements=16L)
    let canonical = Bytes::from_array(
      FixedArray::makei(pack.element_count().to_int() * 4, i => i.to_byte()),
    )
    let packed = Bytes::from_array(
      FixedArray::makei(canonical.length(), i => {
        let source = pack.source_index((i / 4).to_int64()).to_int()
        canonical[source * 4 + i % 4]
      }),
    )
    assert_eq(unpack_joint_latent(packed, pack.inverse()), canonical)
    for
      malformed in [b"", b"\x00", canonical[:canonical.length() - 4].to_owned()] {
      try unpack_joint_latent(malformed, pack.inverse()) catch {
        _ => ()
      } noraise {
        _ => fail("incomplete terminal tensor accepted")
      }
    }
  }
}
