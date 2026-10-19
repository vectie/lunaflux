///|
test "decoded-piece bound follows actual vocabulary rather than admission ceiling" {
  let spec = luna_work_corpus_spec()
  assert_eq(spec.maximum_decoded_piece_bytes(), 5)
  assert_true(
    spec.maximum_decoded_piece_bytes() < spec.limits().max_token_bytes,
  )
  for id in 0..<spec.vocab_size() {
    for policy in [Skip, Preserve] {
      assert_true(
        spec.decoded_piece_bytes(id, special_policy=policy) <=
        spec.maximum_decoded_piece_bytes(),
      )
    }
  }
}
