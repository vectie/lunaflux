///|
test "buffered dense lowering preserves decode and admits scalar tails" {
  for (rows, columns, inner) in [(64, 4096, 4096), (17, 65, 33), (17, 65, 136)] {
    let plan = @matrix.MatrixTilePlan::new(
      rows~,
      columns~,
      inner~,
      law=ShapeAwareFloat32,
      staging=Buffered(inner_tile=64),
    )
    let text = @utf8.decode(
      @tiled_matrix_cuda_source.dense_source(plan, symbol="buffered"),
    )
    assert_eq(plan.operand_slots(), 2)
    assert_eq(plan.tile(), (16, 64, 64))
    assert_true(text.contains("if(live_rows==1)"))
    assert_true(text.contains("publish next slot and retire current readers"))
    assert_true(text.contains("kk+=16"))
    assert_eq(text.contains("cp.async.cg.shared.global"), inner % 8 == 0)
    assert_false(text.contains("mbarrier.try_wait"))
  }
}

///|
test "expression-only operands cannot silently discard buffered staging" {
  let plan = @matrix.MatrixTilePlan::new(
    rows=16,
    columns=64,
    inner=32,
    law=ShapeAwareFloat32,
    staging=Buffered(inner_tile=32),
  )
  try
    @tiled_matrix_cuda_source.matrix_tile_body(
      plan,
      input="left",
      weight="right",
      result="output",
    )
  catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    _ => fail("unexpected lowering error")
  } noraise {
    _ => fail("address-free operand accepted an unimplemented copy schedule")
  }
  try
    @tiled_matrix_cuda_source.paired_matrix_tile_body(
      plan,
      input="left",
      weights=("right0", "right1"),
      results=("out0", "out1"),
    )
  catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    _ => fail("unexpected paired lowering error")
  } noraise {
    _ => fail("paired expressions silently ignored copy schedule")
  }
}

///|
test "export buffered matrix paired probe" {
  if @env.get_env_var("LUNA_EXPORT_BUFFERED_MATRIX") != Some("1") {
    return
  }
  println("LUNA_SOURCE_BEGIN")
  let tile = match @env.get_env_var("LUNA_BUFFERED_K") {
    Some("32") => 32
    Some("128") => 128
    _ => 64
  }
  for
    i, (rows, columns, inner) in [(64, 4096, 4096), (17, 65, 33), (17, 65, 136)] {
    for
      (label, staging) in [
        ("old", @matrix.Synchronous),
        ("next", Buffered(inner_tile=tile)),
      ] {
      let plan = @matrix.MatrixTilePlan::new(
        rows~,
        columns~,
        inner~,
        law=ShapeAwareFloat32,
        staging~,
      )
      println(
        @utf8.decode(
          @tiled_matrix_cuda_source.dense_source(
            plan,
            symbol="matrix_\{label}_\{i}",
          ),
        ),
      )
      println("static constexpr int grid_\{label}_\{i}=\{plan.tile_count()};")
    }
  }
  println("LUNA_SOURCE_END")
}
