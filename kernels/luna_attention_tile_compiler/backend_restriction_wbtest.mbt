///|
test "backend availability filtering retains resource feedback and rejects an infeasible sole candidate" {
  let base = transfer_frontier_request([])
  let frontier = compile_attention_tile_frontier(base)
  let winner = frontier.selected()
  let changed = base.with_resource_feedback(feedback_test_budget(), [
    AttentionRegisterObservation::new(
      winner.schedule().digest(),
      registers_per_thread=1024,
    ),
  ])
  let rescored = compile_attention_tile_frontier(changed)
  let ids = frontier
    .variants()
    .map(variant => variant.plan().candidate().stable_id())
  let restricted = rescored.restrict_to(changed, ids[:])
  assert_true(
    restricted.selected().schedule().digest() != winner.schedule().digest(),
  )
  assert_eq(restricted.selected(), rescored.selected())
  try
    rescored.restrict_to(changed, [winner.plan().candidate().stable_id()])
  catch {
    InvalidAttentionTileCompilation(ResourceFeedback) => ()
    error => raise error
  } noraise {
    _ => fail("backend filter bypassed infeasible register feedback")
  }
  assert_true(
    try frontier.restrict_to(base, []) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
