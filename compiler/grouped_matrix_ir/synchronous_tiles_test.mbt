///|
test "synchronous staging windows preserve output ownership and numeric law" {
  let baseline = @grouped_matrix_ir.MatrixTilePlan::new(
    rows=64,
    columns=4096,
    inner=4096,
    law=ShapeAwareFloat32,
  )
  for k in [32, 64, 128] {
    let plan = @grouped_matrix_ir.MatrixTilePlan::new(
      rows=64,
      columns=4096,
      inner=4096,
      law=ShapeAwareFloat32,
      staging=SynchronousTiles(inner_tile=k),
    )
    assert_eq(plan.tile(), (16, 64, k))
    assert_eq(plan.tile_count(), baseline.tile_count())
    assert_eq(plan.law(), baseline.law())
    assert_eq(plan.operand_slots(), 1)
    let segments = @grouped_matrix_ir.SegmentedMatrixPlan::new([plan, plan])
    assert_eq(segments.scratch_elements(), (16 * k, 64 * k, 1024))
  }
  for invalid in [0, 16, 48, 256] {
    try
      @grouped_matrix_ir.MatrixTilePlan::new(
        rows=64,
        columns=4096,
        inner=4096,
        law=ShapeAwareFloat32,
        staging=SynchronousTiles(inner_tile=invalid),
      )
    catch {
      @precision.PrecisionPlanError::NoImplementation => ()
      _ => fail("unexpected invalid-window error")
    } noraise {
      _ => fail("unsupported synchronous window")
    }
  }
}

///|
test "workset budget resolves a concrete plan without changing large matrices" {
  for
    (n, k, want) in [
      (1536, 4096, 128),
      (16384, 512, 32),
      (8192, 1536, 32),
      (4096, 4096, 32),
      (4096, 1536, 32),
      (65, 33, 32),
      (65, 136, 32),
    ] {
    let p = @grouped_matrix_ir.MatrixTilePlan::new(
      rows=64,
      columns=n,
      inner=k,
      law=ShapeAwareFloat32,
      staging=WorksetBoundedTiles(
        inner_tile=128,
        maximum_weight_bytes=16777216L,
        maximum_tiles=192,
      ),
    )
    assert_eq(p.tile(), (16, 64, want))
    assert_eq(p.operand_slots(), 1)
    assert_false(p.staging() is WorksetBoundedTiles(..))
  }
}
