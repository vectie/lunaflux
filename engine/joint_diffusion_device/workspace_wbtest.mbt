///|
test "block arena separates live hidden tensors and shares attention MLP scratch" {
  let plan = BlockWorkspaceLayout::new(
    rows=3,
    hidden_width=4,
    head_count=2,
    head_dimension=4,
    intermediate_width=100,
    max_bytes=1112L,
  )
  assert_eq(plan.hidden_bytes, 24L)
  assert_eq(plan.normalized_offset, 0L)
  assert_eq(plan.branch_offset, 256L)
  assert_eq(plan.scratch_offset, 512L)
  assert_eq(plan.scratch_bytes, 600L)
  assert_eq(plan.byte_count, 1112L)
  let attention_dominant = BlockWorkspaceLayout::new(
    rows=3,
    hidden_width=4,
    head_count=2,
    head_dimension=4,
    intermediate_width=7,
    max_bytes=704L,
  )
  assert_eq(attention_dominant.scratch_bytes, 192L)
  assert_eq(attention_dominant.byte_count, 704L)
  try
    BlockWorkspaceLayout::new(
      rows=3,
      hidden_width=4,
      head_count=2,
      head_dimension=4,
      intermediate_width=100,
      max_bytes=1111L,
    )
  catch {
    @device.DeviceError::InvalidArgument(AllocationCreate) => ()
    _ => fail("wrong workspace budget error")
  } noraise {
    _ => fail("over-budget workspace admitted")
  }
}
