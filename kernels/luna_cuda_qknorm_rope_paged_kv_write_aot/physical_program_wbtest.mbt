///|
test "ingress physical migration preserves source" {
  let candidate = postprocess_test_candidate(postprocess_test_fixture())
  assert_eq(candidate.physical_program().geometry(), (8, 6, 256))
  assert_eq(candidate.physical_numerics().shape(), (128, 32))
  assert_eq(candidate.physical_head_map().geometry(), (2, 4, 32))
  inspect(
    candidate.source_digest().as_hex(),
    content=(
      #|92efd3cf7f465cadd0003f151eecab2eac538fc7161eb516f24a4c6fd92998cf
    ),
  )
}
