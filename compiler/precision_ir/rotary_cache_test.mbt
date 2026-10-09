///|
test "standalone compact cache transform counts one tensor and exact prefix blocks" {
  let plan = @precision_ir.RotaryCachePrecision::new(
    head_count=1,
    head_width=512,
    rotary_width=64,
    maximum_rows=2,
    maximum_positions=1048576,
    frequency=Base(theta=10000.0),
    simulation_block_width=64,
    simulation_amax_floor=0.0001,
    simulation_scale=PowerOfTwo,
    maximum_workspace_bytes=2048L,
  )
  assert_eq(plan.geometry(), (1, 512, 64, 2, 1048576))
  assert_eq(plan.workspace_bytes(), 2048L)
  assert_eq(
    plan.simulation(),
    (64, 0.0001, @precision_ir.SimulatedCacheScale::PowerOfTwo),
  )
  assert_eq(plan.effect(), @precision_ir.NumericEffect::StepWrite)
  for
    (width, rotary, block, budget) in [
      (512, 64, 64, 2047L),
      (512, 64, 128, 2048L),
      (512, 65, 64, 2048L),
      (512, 64, 0, 2048L),
    ] {
    try
      @precision_ir.RotaryCachePrecision::new(
        head_count=1,
        head_width=width,
        rotary_width=rotary,
        maximum_rows=2,
        maximum_positions=1048576,
        frequency=Base(theta=10000.0),
        simulation_block_width=block,
        simulation_amax_floor=0.0001,
        simulation_scale=PowerOfTwo,
        maximum_workspace_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("illegal compact transform accepted")
    }
  }
}
