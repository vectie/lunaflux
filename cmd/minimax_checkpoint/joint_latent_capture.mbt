///|
// Pure diagnostic unpacking. Preserve each F32 bit pattern and the planner's
// canonical ownership; do not invent a second media-layout convention.
fn unpack_joint_latent(
  packed : Bytes,
  permutation : @permutation.Permutation,
) -> Bytes raise {
  let elements = packed.length() / 4
  guard elements > 0 &&
    packed.length() % 4 == 0 &&
    permutation.element_count() == elements.to_int64() else {
    fail("latent capture extent disagrees with the boundary plan")
  }
  Bytes::from_array(
    FixedArray::makei(packed.length(), i => {
      let source = permutation.source_index((i / 4).to_int64()).to_int()
      packed[source * 4 + i % 4]
    }),
  )
}
