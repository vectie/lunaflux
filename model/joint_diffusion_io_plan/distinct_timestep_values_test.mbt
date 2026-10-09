///|
test "distinct timestep values bind exact F32 bits and selection identity" {
  let (request, timestep, _, _) = modality_fixture()
  let values = [0.0F, 0.5F, 1.0F]
  let first = JointDistinctTimestepValues::new(request, timestep, values~)
  values[1] = 0.25F
  assert_eq(first.values(), ReadOnlyArray::from_array([0.0F, 0.5F, 1.0F]))
  let repeated = JointDistinctTimestepValues::new(request, timestep, values=[
    0.0F, 0.5F, 1.0F,
  ])
  assert_eq(first.digest(), repeated.digest())
}

///|
test "distinct timestep values reject count nonfinite and range attacks" {
  let (request, timestep, _, _) = modality_fixture()
  for values in [[0.0F], [0.0F, -0.1F, 1.0F], [0.0F, 1.1F, 1.0F]] {
    try JointDistinctTimestepValues::new(request, timestep, values~) catch {
      InvalidDistinctTimestepValues => ()
      _ => fail("unexpected timestep-value rejection")
    } noraise {
      _ => fail("hostile timestep values admitted")
    }
  }
}
