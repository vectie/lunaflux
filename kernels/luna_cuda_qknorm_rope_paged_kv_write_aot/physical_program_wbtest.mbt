///|
test "ingress physical migration preserves source" {
  let candidate = postprocess_test_candidate(postprocess_test_fixture())
  assert_eq(candidate.physical_program().geometry(), (8, 6, 256))
  assert_eq(candidate.physical_numerics().shape(), (128, 32))
  assert_eq(candidate.physical_head_map().geometry(), (2, 4, 32))
  inspect(
    candidate.source_digest().as_hex(),
    content=(
      #|9650bc08a6a03daa7476d71954e27147976fc5001a75de9c645995b27d38c3c7
    ),
  )
}
