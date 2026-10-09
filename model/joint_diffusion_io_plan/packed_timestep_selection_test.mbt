///|
test "packed timestep selection is immutable bounded and canonical" {
  let request = io_text_request(io_minimax_plan())
  let indices = [0, 1, 2, 1]
  let first = JointPackedTimestepSelection::new(
    request,
    evaluation_index=7,
    distinct_timestep_count=3,
    indices~,
  )
  indices[0] = 2
  assert_eq(first.evaluation_index(), 7)
  assert_eq(first.packed_row_count(), 4)
  assert_eq(first.distinct_timestep_count(), 3)
  assert_eq(first.indices(), ReadOnlyArray::from_array([0, 1, 2, 1]))
  let repeated = JointPackedTimestepSelection::new(
    request,
    evaluation_index=7,
    distinct_timestep_count=3,
    indices=[0, 1, 2, 1],
  )
  assert_eq(first.digest(), repeated.digest())
  let changed = JointPackedTimestepSelection::new(
    request,
    evaluation_index=8,
    distinct_timestep_count=3,
    indices=[0, 1, 2, 1],
  )
  assert_true(first.digest() != changed.digest())
}

///|
test "packed timestep selection rejects hostile evaluation rows and indices" {
  let request = io_text_request(io_minimax_plan())
  for
    hostile in [
      (-1, 1, [0]),
      (50, 1, [0]),
      (0, 0, [0]),
      (0, 2, [-1]),
      (0, 2, [2]),
      (0, 1, []),
    ] {
    let (evaluation, distinct, indices) = hostile
    try
      JointPackedTimestepSelection::new(
        request,
        evaluation_index=evaluation,
        distinct_timestep_count=distinct,
        indices~,
      )
    catch {
      InvalidPackedTimestepSelection => ()
      _ => fail("unexpected packed timestep rejection")
    } noraise {
      _ => fail("hostile packed timestep selection admitted")
    }
  }
}
