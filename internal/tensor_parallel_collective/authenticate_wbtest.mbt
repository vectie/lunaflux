///|
test "logical bridge derives deterministic rank-specific contract identity" {
  let (model, bindings) = bridge_fixture()
  let rank = bridge_rank_plan(model, bindings, 1, 2)
  let first = authenticate_rank_plan(
    rank,
    expected_identity=model.identity(),
    max_query_tokens=32,
  )
  let replay = authenticate_rank_plan(
    rank,
    expected_identity=model.identity(),
    max_query_tokens=32,
  )
  assert_eq(first.digest(), replay.digest())
  assert_eq(first.rank(), 1)
  assert_eq(first.world_size(), 2)
  assert_eq(first.site_count(), rank.collective_count())
  let left = authenticate_rank_plan(
    bridge_rank_plan(model, bindings, 0, 2),
    expected_identity=model.identity(),
    max_query_tokens=32,
  )
  assert_true(first.digest() != left.digest())
}

///|
test "logical bridge rejects model identity substitution and single rank plans" {
  let (model, bindings) = bridge_fixture()
  let rank = bridge_rank_plan(model, bindings, 0, 2)
  let (foreign, _) = bridge_fixture(
    content="1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  try
    authenticate_rank_plan(
      rank,
      expected_identity=foreign.identity(),
      max_query_tokens=32,
    )
  catch {
    InvalidRankPlan(RankPlanIdentity) => ()
    _ => fail("unexpected identity substitution failure")
  } noraise {
    _ => fail("foreign model identity was admitted")
  }
  try
    authenticate_rank_plan(
      bridge_rank_plan(model, bindings, 0, 1),
      expected_identity=model.identity(),
      max_query_tokens=32,
    )
  catch {
    InvalidRankPlan(RankPlanShape) => ()
    _ => fail("unexpected single-rank failure")
  } noraise {
    _ => fail("single-rank no-collective plan was admitted")
  }
}

///|
test "logical bridge binds and validates the maximum query-token count" {
  let (model, bindings) = bridge_fixture()
  let rank = bridge_rank_plan(model, bindings, 0, 2)
  let small = authenticate_rank_plan(
    rank,
    expected_identity=model.identity(),
    max_query_tokens=1,
  )
  let larger = authenticate_rank_plan(
    rank,
    expected_identity=model.identity(),
    max_query_tokens=32,
  )
  assert_true(small.digest() != larger.digest())
  try
    authenticate_rank_plan(
      rank,
      expected_identity=model.identity(),
      max_query_tokens=0,
    )
  catch {
    InvalidRankPlan(RankPlanShape) => ()
    _ => fail("unexpected maximum-query-token failure")
  } noraise {
    _ => fail("zero maximum query tokens were admitted")
  }
}
