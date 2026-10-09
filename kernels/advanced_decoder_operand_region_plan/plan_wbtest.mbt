///|
fn operand_region_test_operand(
  role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
  byte_count : Int64,
) -> @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand {
  try! @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand::new(
    role~,
    byte_count~,
    alignment=4L,
  )
}

///|
fn operand_region_test_input(
  ordinal : Int,
  role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
  fill? : Char = 'a',
  offset~ : Int64,
  capacity? : Int64 = 256L,
) -> AdvancedDecoderOperandRegionInput {
  try! AdvancedDecoderOperandRegionInput::new(
    operand_ordinal=ordinal,
    role~,
    allocation_identity_sha256=String::make(64, fill),
    device_offset=offset,
    allocation_capacity_bytes=capacity,
  )
}

///|
fn operand_region_test_limits(
  canonical? : Int = 4096,
) -> AdvancedDecoderOperandRegionPlanLimits {
  try! AdvancedDecoderOperandRegionPlanLimits::new(
    maximum_operands=8,
    maximum_allocation_bytes=4096L,
    maximum_canonical_bytes=canonical,
  )
}

///|
fn operand_region_test_operands() -> Array[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  [
    operand_region_test_operand(StepCountsInput, 20L),
    operand_region_test_operand(TokenIdsInput, 8L),
    operand_region_test_operand(TokenHashTableInput, 64L),
    operand_region_test_operand(RoutingIndicesOutput, 48L),
  ]
}

///|
test "exact allocation-relative regions produce deterministic inert plan" {
  let operands = operand_region_test_operands()
  let inputs = [
    operand_region_test_input(0, StepCountsInput, offset=0L),
    operand_region_test_input(1, TokenIdsInput, offset=20L),
    operand_region_test_input(2, TokenHashTableInput, fill='b', offset=0L),
    operand_region_test_input(3, RoutingIndicesOutput, offset=32L),
  ]
  let first = plan_advanced_decoder_operand_regions(
    operands,
    inputs,
    operand_region_test_limits(),
  )
  let repeated = plan_advanced_decoder_operand_regions(
    operands,
    inputs,
    operand_region_test_limits(),
  )
  assert_eq(first.status(), InertUnqualified)
  assert_eq(first.regions().length(), 4)
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.digest().as_hex().length(), 64)
  guard first.region(3) is Some(output) else {
    fail("output operand region missing")
  }
  assert_eq(output.operand_ordinal(), 3)
  assert_eq(output.role(), RoutingIndicesOutput)
  assert_eq(output.byte_count(), 48L)
  assert_eq(output.alignment(), 4L)
  assert_eq(output.device_offset(), 32L)
  assert_eq(output.end_offset(), 80L)
  assert_eq(output.allocation_capacity_bytes(), 256L)
  try first.require_launch_authority() catch {
    LaunchAuthorityUnavailable => ()
    _ => fail("unexpected launch-authority rejection")
  } noraise {
    _ => fail("inert operand regions gained launch authority")
  }
}

///|
test "wrong ordinal and role reject exact ABI planning" {
  let operands = operand_region_test_operands()
  let hostile = [
    [
      operand_region_test_input(1, StepCountsInput, offset=0L),
      operand_region_test_input(1, TokenIdsInput, offset=20L),
      operand_region_test_input(2, TokenHashTableInput, fill='b', offset=0L),
      operand_region_test_input(3, RoutingIndicesOutput, offset=32L),
    ],
    [
      operand_region_test_input(0, TokenIdsInput, offset=0L),
      operand_region_test_input(1, TokenIdsInput, offset=20L),
      operand_region_test_input(2, TokenHashTableInput, fill='b', offset=0L),
      operand_region_test_input(3, RoutingIndicesOutput, offset=32L),
    ],
  ]
  for index, inputs in hostile {
    try
      plan_advanced_decoder_operand_regions(
        operands,
        inputs,
        operand_region_test_limits(),
      )
    catch {
      OperandOrdinalMismatch(0) if index == 0 => ()
      OperandRoleMismatch(0) if index == 1 => ()
      _ => fail("unexpected ordinal/role rejection")
    } noraise {
      _ => fail("wrong operand ordinal or role was planned")
    }
  }
}

///|
test "misalignment and undersized capacity fail closed" {
  let operands = operand_region_test_operands()
  let hostile = [
    [
      operand_region_test_input(0, StepCountsInput, offset=2L),
      operand_region_test_input(1, TokenIdsInput, offset=20L),
      operand_region_test_input(2, TokenHashTableInput, fill='b', offset=0L),
      operand_region_test_input(3, RoutingIndicesOutput, offset=32L),
    ],
    [
      operand_region_test_input(0, StepCountsInput, offset=0L),
      operand_region_test_input(1, TokenIdsInput, offset=252L),
      operand_region_test_input(2, TokenHashTableInput, fill='b', offset=0L),
      operand_region_test_input(3, RoutingIndicesOutput, offset=32L),
    ],
  ]
  for index, inputs in hostile {
    try
      plan_advanced_decoder_operand_regions(
        operands,
        inputs,
        operand_region_test_limits(),
      )
    catch {
      OperandAlignmentMismatch(0) if index == 0 => ()
      RegionUndersized(1) if index == 1 => ()
      _ => fail("unexpected alignment/capacity rejection")
    } noraise {
      _ => fail("misaligned or undersized region was planned")
    }
  }
}

///|
test "same-allocation overlap and capacity drift fail closed" {
  let operands = operand_region_test_operands()
  let hostile = [
    [
      operand_region_test_input(0, StepCountsInput, offset=0L),
      operand_region_test_input(1, TokenIdsInput, offset=16L),
      operand_region_test_input(2, TokenHashTableInput, fill='b', offset=0L),
      operand_region_test_input(3, RoutingIndicesOutput, offset=32L),
    ],
    [
      operand_region_test_input(0, StepCountsInput, offset=0L),
      operand_region_test_input(1, TokenIdsInput, offset=20L),
      operand_region_test_input(2, TokenHashTableInput, fill='b', offset=0L),
      operand_region_test_input(
        3,
        RoutingIndicesOutput,
        offset=32L,
        capacity=128L,
      ),
    ],
  ]
  for index, inputs in hostile {
    try
      plan_advanced_decoder_operand_regions(
        operands,
        inputs,
        operand_region_test_limits(),
      )
    catch {
      RegionOverlap(0, 1) if index == 0 => ()
      AllocationCapacityMismatch(3) if index == 1 => ()
      _ => fail("unexpected overlap/capacity rejection")
    } noraise {
      _ => fail("overlap or allocation-capacity drift was planned")
    }
  }
}
