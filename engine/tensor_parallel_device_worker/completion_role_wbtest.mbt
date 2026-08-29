///|
test "rank zero alone owns canonical completion publication" {
  assert_true(completion_role(0) == LeaderRole)
  for rank in 1..<16 {
    assert_true(completion_role(rank) == FollowerRole)
  }
}

///|
test "follower acknowledgement retains exact scalar transcript" {
  let ack : TensorParallelFollowerExecutionAck = {
    group_generation_value: 9UL,
    model_generation_value: 4UL,
    plan_sequence_value: 21UL,
    rank: 3,
    world_size: 4,
  }
  assert_eq(ack.group_generation_value(), 9UL)
  assert_eq(ack.model_generation_value(), 4UL)
  assert_eq(ack.plan_sequence_value(), 21UL)
  assert_eq(ack.rank(), 3)
  assert_eq(ack.world_size(), 4)
}

///|
test "leader row count admission returns one scalar for every legal split" {
  assert_eq(try! require_leader_row_counts(4, 4, 0), 4)
  assert_eq(try! require_leader_row_counts(4, 2, 2), 2)
  assert_eq(try! require_leader_row_counts(4, 0, 4), 0)
}

///|
test "leader row count admission rejects hostile and substituted totals" {
  for counts in [(-1, 0, 0), (4, -1, 5), (4, 1, -1), (4, 2, 1), (4, 5, 0)] {
    try require_leader_row_counts(counts.0, counts.1, counts.2) catch {
      Invalid(PlanFrame) => ()
      _ => fail("unexpected leader row count error")
    } noraise {
      _ => fail("invalid leader row counts were admitted")
    }
  }
}
