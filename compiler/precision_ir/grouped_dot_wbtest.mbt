///|
test "grouped dot validates exact order rather than device geometry" {
  let law : DotReductionLaw = GroupedVectorFloat32(
    groups=18,
    partitions=128,
    vector_width=2,
    publication_partitions=4,
    multiply_add=FusedMultiplyAdd,
  )
  law.validate(16384)
  law.validate(17) // Tails and empty groups have an explicit zero identity.
  for
    (groups, partitions, vector_width, publication_partitions) in [
      (0, 128, 2, 4),
      (18, 3, 2, 4),
      (18, 128, 3, 4),
      (18, 128, 2, 0),
      (18, 128, 2, 19),
      (2147483647, 128, 2, 4),
    ] {
    try
      GroupedVectorFloat32(
        groups~,
        partitions~,
        vector_width~,
        publication_partitions~,
        multiply_add=FusedMultiplyAdd,
      ).validate(16384)
    catch {
      IllegalPlacement => ()
      _ => fail("unexpected invalid order error")
    } noraise {
      _ => fail("invalid grouped order accepted")
    }
  }
}
