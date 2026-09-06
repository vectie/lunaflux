///|
test "selected row materialization has an explicit storage cost" {
  let shape = ProjectionShape::new(
    LanguageModelHead,
    input_width=1024,
    output_width=151936,
  )
  let caps = ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile=Some(ProjectionTile::new(rows=16, columns=16, reduction=16)),
  )
  let streamed = plan_projection_dispatch(shape, 256, caps).prefill()
  let resident = streamed.with_selected_row_residency(stable_id=3008)
  assert_eq(streamed.local_storage(shape).accumulator_elements(), 2048L)
  assert_eq(streamed.local_storage(shape).selected_input_elements(), 256L)
  assert_eq(resident.local_storage(shape).accumulator_elements(), 2048L)
  assert_eq(resident.local_storage(shape).selected_input_elements(), 16384L)
  assert_eq(resident.tile(), streamed.tile())
}

///|
test "independent output partitions retain residency and adjust local cost" {
  let shape = ProjectionShape::new(
    LanguageModelHead,
    input_width=2048,
    output_width=32768,
  )
  let caps = ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile=Some(ProjectionTile::new(rows=16, columns=16, reduction=16)),
  )
  let resident = plan_projection_dispatch(shape, 32, caps)
    .prefill()
    .with_selected_row_residency(stable_id=3008)
  for candidate in resident.output_distribution_candidates() {
    assert_eq(candidate.input_residency(), SelectedRowTileResidency)
    assert_eq(candidate.local_storage(shape).selected_input_elements(), 32768L)
    assert_eq(
      candidate.local_storage(shape).accumulator_elements(),
      candidate.output_tiles_per_workgroup().to_int64() * 256L,
    )
  }
}
