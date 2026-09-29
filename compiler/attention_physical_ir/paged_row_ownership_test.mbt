///|
test "paged address ownership preserves each row and subgroup" {
  for subgroup in [8, 16, 32, 64] {
    for width in [1, 2, 4, 8] {
      for row in [32, 64, 96, 128, 256] {
        let plan = @attention_physical_ir.PagedRowOwnership::new(
          row_elements=row,
          vector_elements=width,
          subgroup_size=subgroup,
          hoist=true,
        ).unwrap()
        for owner in 0..<(subgroup * 3) {
          let source = plan.address_owner(owner)
          assert_eq(source / plan.consumers(), owner / plan.consumers())
          assert_eq(source / subgroup, owner / subgroup)
          assert_true(source <= owner)
        }
      }
    }
  }
  assert_true(
    @attention_physical_ir.PagedRowOwnership::new(
      row_elements=128,
      vector_elements=4,
      subgroup_size=32,
      hoist=true,
    )
    .unwrap()
    .shares_address(),
  )
  assert_false(
    @attention_physical_ir.PagedRowOwnership::new(
      row_elements=128,
      vector_elements=4,
      subgroup_size=32,
      hoist=false,
    )
    .unwrap()
    .shares_address(),
  )
  assert_true(
    @attention_physical_ir.PagedRowOwnership::new(
      row_elements=127,
      vector_elements=4,
      subgroup_size=32,
      hoist=true,
    )
    is None,
  )
}
