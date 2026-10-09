///|
test "head reduction fixture cannot bypass earlier Query-B projection gap" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let ordinal = 9
  assert_true(
    requirements.requirements()[ordinal].operation()
    is HyperConnectionHeadReduceV1(2, _, _),
  )
  let launches = artifact_test_launches(requirements, target)
  let operands = artifact_test_operands(
    requirements.requirements()[ordinal].operation(),
  )
  assert_eq(operands.length(), 6)
  assert_eq(operands[0].role(), StepCountsInput)
  assert_eq(operands[0].byte_count(), 20L)
  assert_eq(operands[1].role(), HyperConnectionStateInput)
  assert_eq(operands[1].byte_count(), 512L)
  assert_eq(operands[2].role(), HyperConnectionFunctionWeightsInput)
  assert_eq(operands[2].byte_count(), 2048L)
  assert_eq(operands[3].role(), HyperConnectionBaseInput)
  assert_eq(operands[3].byte_count(), 8L)
  assert_eq(operands[4].role(), HyperConnectionScaleInput)
  assert_eq(operands[4].byte_count(), 4L)
  assert_eq(operands[5].role(), HyperConnectionReducedOutput)
  assert_eq(operands[5].byte_count(), 256L)
  operands[2] = try! AdvancedDecoderLaunchOperand::new(
    role=ModelWeights,
    byte_count=2048L,
    alignment=16L,
  )
  launches[ordinal] = artifact_test_launch(
    requirements,
    target,
    ordinal,
    operands~,
  )
  try artifact_test_admit(requirements, target, launches) catch {
    UnsupportedOperation(actual) => assert_eq(actual, 4)
    _ => fail("unexpected earlier Query-B projection rejection")
  } noraise {
    _ => fail("unproven Query-B projection was bypassed")
  }
}

///|
test "truncated head fixture cannot bypass earlier Query-B projection gap" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let ordinal = 9
  let launches = artifact_test_launches(requirements, target)
  let operands = artifact_test_operands(
    requirements.requirements()[ordinal].operation(),
  )
  operands[5] = try! AdvancedDecoderLaunchOperand::new(
    role=HyperConnectionReducedOutput,
    byte_count=254L,
    alignment=16L,
  )
  launches[ordinal] = artifact_test_launch(
    requirements,
    target,
    ordinal,
    operands~,
  )
  try artifact_test_admit(requirements, target, launches) catch {
    UnsupportedOperation(actual) => assert_eq(actual, 4)
    _ => fail("unexpected earlier Query-B projection rejection")
  } noraise {
    _ => fail("unproven Query-B projection was bypassed")
  }
}
