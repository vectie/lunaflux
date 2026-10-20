///|
test "row reuse covers tails and shares only compatible scratch" {
  for (rows, columns) in [(3, 65), (17, 64), (33, 65), (64, 4096)] {
    let p = @grouped_matrix_ir.MatrixTilePlan::new(
      rows~,
      columns~,
      inner=33,
      law=MatrixFloat32,
      tile_rows=32,
      tile_columns=32,
    )
    assert_eq(p.tile_count(), (rows + 31) / 32 * ((columns + 31) / 32))
    let segments = @grouped_matrix_ir.SegmentedMatrixPlan::new([p, p])
    assert_eq(segments.scratch_elements(), (1024, 1024, 1024))
    let old = @grouped_matrix_ir.MatrixTilePlan::new(
      rows~,
      columns~,
      inner=33,
      law=MatrixFloat32,
    )
    try @grouped_matrix_ir.SegmentedMatrixPlan::new([p, old]) catch {
      @precision.PrecisionPlanError::IllegalPlacement => ()
      _ => fail("wrong scratch mismatch error")
    } noraise {
      _ => fail("mixed ownership admitted")
    }
  }
}

///|
test "unsupported ownership and copy combinations reject" {
  for
    (m, n, staging) in [
      (0, 32, @grouped_matrix_ir.Synchronous),
      (64, 16, Synchronous),
      (32, 64, Synchronous),
      (32, 32, Buffered(inner_tile=32)),
    ] {
    try
      @grouped_matrix_ir.MatrixTilePlan::new(
        rows=64,
        columns=4096,
        inner=4096,
        law=ShapeAwareFloat32,
        tile_rows=m,
        tile_columns=n,
        staging~,
      )
    catch {
      @precision.PrecisionPlanError::NoImplementation => ()
      _ => fail("wrong unsupported schedule error")
    } noraise {
      _ => fail("unsupported schedule accepted")
    }
  }
}

///|
test "supplied reuse envelope preserves short reductions" {
  for k in [128, 1536, 4096, 8192] {
    let p = @grouped_matrix_ir.MatrixTilePlan::new(
      rows=64,
      columns=4096,
      inner=k,
      law=ShapeAwareFloat32,
      tile_rows=32,
      tile_columns=32,
      staging=SynchronousTiles(inner_tile=64),
      minimum_row_reuse_inner=4096,
    )
    assert_eq(p.tile(), if k < 4096 { (16, 64, 32) } else { (32, 32, 64) })
  }
}
