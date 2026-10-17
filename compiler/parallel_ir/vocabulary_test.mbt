///|
test "vocabulary ownership is a complete disjoint uneven cover" {
  let shards = [0, 1, 2].map(rank => {
    VocabularyShard::new(vocabulary=17, rank~, world=3, maximum_outputs=2)
  })
  assert_eq(shards.map(s => s.columns()), [(0, 5), (5, 6), (11, 6)])
  assert_eq(shards[0].candidate_width(), 6)
  assert_true(
    (try? VocabularyShard::new(
      vocabulary=16777217,
      rank=0,
      world=2,
      maximum_outputs=1,
    ))
    is Err(_),
  )
  assert_true(
    (try? VocabularyShard::new(vocabulary=1, rank=0, world=2, maximum_outputs=1))
    is Err(_),
  )
}
