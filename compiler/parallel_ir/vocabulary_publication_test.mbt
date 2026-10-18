///|
test "full logits preserve uneven token domains with equal collective slots" {
  let shards = [0, 1, 2].map(rank => {
    VocabularyShard::new(
      vocabulary=17,
      rank~,
      world=3,
      maximum_outputs=2,
      publication=FullLogits,
    )
  })
  assert_eq(shards.map(s => s.columns()), [(0, 5), (5, 6), (11, 6)])
  assert_eq(shards.map(s => s.transfer_width()), [12, 12, 12])
  assert_eq(shards[0].dense_logits_bytes(), 136L)
  assert_eq(shards[0].publication(), FullLogits)
  assert_true(
    (try? VocabularyShard::new(
      vocabulary=16777216,
      rank=0,
      world=2,
      maximum_outputs=1024,
      publication=FullLogits,
    ))
    is Err(_),
  )
}
