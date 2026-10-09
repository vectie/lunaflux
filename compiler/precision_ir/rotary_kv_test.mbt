///|
fn rotary_fixture(
  frequency : RotaryFrequency,
  budget : Int64,
) -> RotaryKvPrecision raise {
  @precision_ir.RotaryKvPrecision::new(
    query_heads=2,
    key_value_heads=1,
    head_width=128,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=1048576,
    frequency~,
    simulation_block_width=64,
    simulation_amax_floor=0.0001,
    simulation_scale=PowerOfTwo,
    maximum_workspace_bytes=budget,
  )
}

///|
test "rotary simulation plans full-head output bytes without changing cache storage" {
  let plan = rotary_fixture(Base(theta=10000.0), 2304L)
  assert_eq(plan.geometry(), (2, 1, 128, 64, 3, 1048576))
  assert_eq(plan.frame_bytes()[:].to_owned(), [1536L, 768L])
  assert_eq(plan.workspace_bytes(), 2304L)
  assert_eq(plan.simulation(), (64, 0.0001, PowerOfTwo))
  assert_eq(plan.effect(), StepWrite)
  let yarn : @precision_ir.RotaryFrequency = Yarn(
    theta=160000.0,
    original_positions=65536,
    factor=16.0,
    beta_fast=32.0,
    beta_slow=1.0,
  )
  assert_eq(rotary_fixture(yarn, 2304L).frequency(), yarn)
  try rotary_fixture(Base(theta=10000.0), 2303L) catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    error => raise error
  } noraise {
    _ => fail("one-byte-short aggregate budget accepted")
  }
}

///|
test "rotary precision refuses malformed geometry and frequency policy" {
  for
    (width, rotary, block, floor) in [
      (128, 63, 64, 0.0001),
      (128, 64, 63, 0.0001),
      (126, 64, 64, 0.0001),
      (128, 64, 64, 0.0),
    ] {
    try
      @precision_ir.RotaryKvPrecision::new(
        query_heads=2,
        key_value_heads=1,
        head_width=width,
        rotary_width=rotary,
        maximum_rows=3,
        maximum_positions=1048576,
        frequency=Base(theta=10000.0),
        simulation_block_width=block,
        simulation_amax_floor=floor,
        simulation_scale=PowerOfTwo,
        maximum_workspace_bytes=2304L,
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      error => raise error
    } noraise {
      _ => fail("malformed rotary simulation accepted")
    }
  }
  let invalid : Array[@precision_ir.RotaryFrequency] = [
    Base(theta=0.0),
    Yarn(
      theta=1.0,
      original_positions=65536,
      factor=16.0,
      beta_fast=32.0,
      beta_slow=1.0,
    ),
    Yarn(
      theta=160000.0,
      original_positions=1048577,
      factor=16.0,
      beta_fast=32.0,
      beta_slow=1.0,
    ),
    Yarn(
      theta=160000.0,
      original_positions=65536,
      factor=0.0,
      beta_fast=32.0,
      beta_slow=1.0,
    ),
  ]
  for frequency in invalid {
    try rotary_fixture(frequency, 2304L) catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      error => raise error
    } noraise {
      _ => fail("invalid frequency policy accepted")
    }
  }
}
