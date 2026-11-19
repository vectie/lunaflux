///|
test "vocabulary ownership is a complete disjoint uneven cover" {
  let shards = [0, 1, 2].map(rank => {
    @parallel_ir.VocabularyShard::new(vocabulary=17, rank~, world=3, maximum_outputs=2)
  })
  assert_eq(shards.map(s => s.columns()), [(0, 5), (5, 6), (11, 6)])
  assert_eq(shards[0].candidate_width(), 6)
  try @parallel_ir.VocabularyShard::new(
      vocabulary=16777217,
      rank=0,
      world=2,
      maximum_outputs=1,
    ) catch { _ => () } noraise { _ => fail("expected invalid input to fail") }
  try @parallel_ir.VocabularyShard::new(vocabulary=1, rank=0, world=2, maximum_outputs=1) catch { _ => () } noraise { _ => fail("expected invalid input to fail") }
}
