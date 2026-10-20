///|
test "fragment pipeline preserves fold and owns disjoint swizzled atoms" {
  let matrix = @grouped_matrix_ir.MatrixTilePlan::new(
    rows=64,
    columns=4096,
    inner=8192,
    law=ShapeAwareFloat32,
  )
  for row_tile in [32, 64] {
    let plan = @grouped_matrix_ir.MatrixFragmentPipeline::new(matrix, row_tile~)
    assert_eq(plan.matrix().law(), ShapeAwareFloat32)
    assert_eq(plan.tile_count(), 1024)
    assert_eq(plan.shared_bytes(), 2 * (row_tile + 64) * 128)
    let seen = Array::make(1024, false)
    for i in 0..<1024 {
      let physical = plan.operand_byte_offset(i)
      assert_false(seen[physical])
      seen[physical] = true
      assert_eq(plan.operand_byte_offset(physical), i)
      assert_eq(physical % 16, i % 16)
    }
    assert_true(seen.all(x => x))
  }
  try
    @grouped_matrix_ir.MatrixFragmentPipeline::new(matrix, row_tile=16)
  catch {
    _ => ()
  } noraise {
    _ => fail("unsupported ownership accepted")
  }
}

///|
test "fragment envelope preserves unrelated shapes and arithmetic" {
  let policy = @grouped_matrix_ir.MatrixRowReuse::new(
    minimum_rows=32,
    minimum_inner=1536,
    wide_inner=8192,
    wide_tile=64,
    fragment_inner=8192,
    fragment_columns=4096,
  )
  for
    (rows, columns, inner) in [
      (64, 4096, 8192),
      (64, 1536, 8192),
      (64, 4096, 4096),
      (16, 4096, 8192),
      (64, 4097, 8193),
    ] {
    let original = @grouped_matrix_ir.MatrixTilePlan::new(
      rows~,
      columns~,
      inner~,
      law=ShapeAwareFloat32,
    )
    let selected = policy.apply(original)
    assert_eq(selected.geometry(), original.geometry())
    assert_eq(selected.law(), original.law())
    let expected = rows >= 32 &&
      columns >= 4096 &&
      inner >= 8192 &&
      inner % 64 == 0
    assert_eq(selected.staging() is FragmentPipeline(_), expected)
    if expected {
      assert_eq(selected.tile(), (32, 64, 64))
      assert_eq(selected.operand_slots(), 2)
      let pipeline = @grouped_matrix_ir.MatrixFragmentPipeline::new(
        selected,
        row_tile=32,
      )
      assert_eq(pipeline.tile_count(), selected.tile_count())
    }
  }
}
