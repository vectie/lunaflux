///|
test "row work cardinality must fit the published launch envelope" {
  let base = source_entry(1, 1, "kernel")
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=192,
    grid_y=1,
    grid_z=1,
    block_x=512,
    block_y=1,
    block_z=1,
    shared_memory_bytes=16384,
  )
  for grid in [None, Some(1), Some(48), Some(192)] {
    let variant = KernelRowVariant::new(
      maximum_token_rows=16,
      function_symbol="rows16",
      dimensions~,
      bounded_work_grid_x=grid,
    )
    validate_row_variants(
      KernelEntryPointInput::new(
        entry_point=base.entry_point,
        function_symbol="kernel",
        row_variants=[variant],
      ),
      32,
    )
    assert_eq(variant.bounded_work_grid_x(), grid)
  }
  for grid in [-1, 0, 193, 2147483647] {
    let input = KernelEntryPointInput::new(
      entry_point=base.entry_point,
      function_symbol="kernel",
      row_variants=[
        KernelRowVariant::new(
          maximum_token_rows=16,
          function_symbol="rows16",
          dimensions~,
          bounded_work_grid_x=Some(grid),
        ),
      ],
    )
    try validate_row_variants(input, 32) catch {
      InvalidRowVariants(_) => ()
      _ => fail("unexpected row work rejection")
    } noraise {
      _ => fail("out-of-envelope work grid accepted")
    }
  }
}
