///|
fn assert_next_artifact_operation_unavailable(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  target : @catalog.DeviceTarget,
  launches : ArrayView[AdvancedDecoderLaunchEvidenceInput],
) -> Unit raise {
  let mut expected = -1
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is QueryLowRankBProjection(_, _, _) {
      expected = ordinal
      break
    }
  }
  assert_true(expected >= 0)
  try artifact_test_admit(requirements, target, launches) catch {
    UnsupportedOperation(actual) => assert_eq(actual, expected)
    _ => fail("unexpected unavailable Query-B projection rejection")
  } noraise {
    _ => fail("an artifact with an unproven Query-B projection was admitted")
  }
}

///|
test "complete artifact admission advances through key/value and rejects Query-B" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  assert_next_artifact_operation_unavailable(
    requirements,
    target,
    artifact_test_launches(requirements, target),
  )
}

///|
test "pre-reduction fixture cannot bypass earlier Query-B projection gap" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let launches = artifact_test_launches(requirements, target)
  let ordinal = 7
  assert_true(
    requirements.requirements()[ordinal].operation()
    is HyperConnectionPreReduceV1(_),
  )
  let operands = artifact_test_operands(
    requirements.requirements()[ordinal].operation(),
  )
  operands[2] = try! AdvancedDecoderLaunchOperand::new(
    role=HyperConnectionPreOutput,
    byte_count=8L,
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
test "post-combine fixture cannot bypass earlier Query-B projection gap" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let launches = artifact_test_launches(requirements, target)
  let ordinal = 8
  assert_true(
    requirements.requirements()[ordinal].operation()
    is HyperConnectionPostCombineV1(_),
  )
  let operands = artifact_test_operands(
    requirements.requirements()[ordinal].operation(),
  )
  operands[4] = try! AdvancedDecoderLaunchOperand::new(
    role=HyperConnectionCombinationOutput,
    byte_count=16L,
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
test "caller qualification metadata cannot bypass the next unsupported operation" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let qualification = try! AdvancedDecoderDeclaredQualificationDigest::from_sha256(
    String::make(64, 'd'),
  )
  assert_next_artifact_operation_unavailable(
    requirements,
    target,
    artifact_test_launches(requirements, target, qualification~),
  )
}

///|
test "block-control fixture cannot bypass earlier Query-B projection gap" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let launches = artifact_test_launches(requirements, target)
  let ordinal = 6
  assert_true(
    requirements.requirements()[ordinal].operation()
    is HyperConnectionBlockControlV1(_, _, _, _),
  )
  let operands = artifact_test_operands(
    requirements.requirements()[ordinal].operation(),
  )
  operands[7] = try! AdvancedDecoderLaunchOperand::new(
    role=HyperConnectionCombinationOutput,
    byte_count=8L,
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
