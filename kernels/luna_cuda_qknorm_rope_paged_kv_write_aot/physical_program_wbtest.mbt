///|
test "ingress physical migration preserves source" {
  let candidate = postprocess_test_candidate(postprocess_test_fixture())
  assert_eq(candidate.physical_program().geometry(), (8, 6, 256))
  assert_eq(candidate.physical_numerics().shape(), (128, 32))
  assert_eq(candidate.physical_head_map().geometry(), (2, 4, 32))
  inspect(
    candidate.source_digest().as_hex(),
    content=(
      #|b9d4314db42de925fb9479ac760f107452ae3ac249f9245583d9a2c1f72686e6
    ),
  )
}
