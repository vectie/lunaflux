///|
test "all three official hash layers plan exact inert operand metadata" {
  let requirements = operand_plan_test_requirements()
  let source = operand_plan_test_source(requirements)
  let layout = operand_plan_test_layout(requirements.source_identity())
  let candidate = operand_plan_test_candidate_evidence(requirements)
  let digests : Array[String] = []
  for layer in 0..<3 {
    let first = plan_from_evidence(
      source, layout, requirements, candidate, layer,
    )
    let repeated = plan_from_evidence(
      source, layout, requirements, candidate, layer,
    )
    assert_eq(first.status(), InertUnqualified)
    assert_eq(first.source_identity(), requirements.source_identity())
    assert_eq(first.requirements_digest(), requirements.digest())
    assert_eq(first.requirement_ordinal(), 21)
    assert_eq(first.layer_index(), layer)
    assert_eq(first.routed_expert_count(), 256)
    assert_eq(first.operand_ordinal(), 2)
    assert_eq(first.operand_role(), TokenHashTableInput)
    assert_eq(first.byte_count(), 3102720L)
    assert_eq(first.alignment(), 4L)
    assert_eq(first.region_ordinal(), layer)
    assert_eq(first.region_tensor_ordinal(), layer)
    assert_eq(first.device_offset(), layer.to_int64() * 3102720L)
    assert_eq(first.sidecar_layout_digest(), layout.digest())
    assert_eq(first.digest(), repeated.digest())
    assert_eq(first.digest().as_hex().length(), 64)
    assert_false(digests.contains(first.digest().as_hex()))
    digests.push(first.digest().as_hex())
    try first.require_launch_authority() catch {
      NoLaunchAuthority => ()
      _ => fail("unexpected launch-authority rejection")
    } noraise {
      _ => fail("inert source candidate gained launch authority")
    }
  }
}

///|
test "all five official profiles retain exact token-hash geometry" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 21, 256),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 21, 256),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 20, 384),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 20, 384),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 21, 256),
  ]
  let digests : Array[String] = []
  for entry in cases {
    let (model_id, fill, expected_ordinal, expected_experts) = entry
    let requirements = operand_plan_test_requirements(model_id~, fill~)
    let plan = plan_from_evidence(
      operand_plan_test_source(requirements),
      operand_plan_test_layout(requirements.source_identity()),
      requirements,
      operand_plan_test_candidate_evidence(requirements),
      2,
    )
    assert_eq(plan.requirement_ordinal(), expected_ordinal)
    assert_eq(plan.routed_expert_count(), expected_experts)
    assert_eq(plan.layer_index(), 2)
    assert_eq(plan.byte_count(), 3102720L)
    assert_false(digests.contains(plan.digest().as_hex()))
    digests.push(plan.digest().as_hex())
  }
}

///|
test "source candidate and layout identities cannot be crossed" {
  let requirements = operand_plan_test_requirements()
  let foreign = operand_plan_test_requirements(
    model_id="deepseek-ai/DeepSeek-V4-Pro",
    fill='2',
  )
  let layout = operand_plan_test_layout(requirements.source_identity())
  try
    plan_from_evidence(
      operand_plan_test_source(foreign),
      layout,
      requirements,
      operand_plan_test_candidate_evidence(requirements),
      0,
    )
  catch {
    SourceIdentityMismatch => ()
    _ => fail("unexpected cross-model source rejection")
  } noraise {
    _ => fail("cross-model token-hash source was admitted")
  }
  try
    plan_from_evidence(
      operand_plan_test_source(requirements),
      operand_plan_test_layout(foreign.source_identity()),
      requirements,
      operand_plan_test_candidate_evidence(requirements),
      0,
    )
  catch {
    LayoutRegionMismatch(-1) => ()
    _ => fail("unexpected cross-model layout rejection")
  } noraise {
    _ => fail("cross-model sidecar layout was admitted")
  }
  try
    plan_from_evidence(
      operand_plan_test_source(requirements),
      layout,
      requirements,
      operand_plan_test_candidate_evidence(foreign),
      0,
    )
  catch {
    RequirementsIdentityMismatch => ()
    _ => fail("unexpected cross-model candidate rejection")
  } noraise {
    _ => fail("cross-model CUDA candidate was admitted")
  }
}

///|
test "requirements digest ordinal operation and capability are authenticated" {
  let requirements = operand_plan_test_requirements()
  let foreign = operand_plan_test_requirements(fill='3')
  let source = operand_plan_test_source(requirements)
  let layout = operand_plan_test_layout(requirements.source_identity())
  let exact = operand_plan_test_candidate_evidence(requirements)
  let hostile = [
    { ..exact, requirements_digest: foreign.digest(), },
    { ..exact, requirement_ordinal: exact.requirement_ordinal + 1, },
    { ..exact, operation: NoAuxiliaryLossTopK(256, 6), },
    { ..exact, capability: StableTopKRoutingAot, },
  ]
  for index, candidate in hostile {
    try plan_from_evidence(source, layout, requirements, candidate, 0) catch {
      RequirementsDigestMismatch if index == 0 => ()
      CandidateRequirementOrdinalMismatch(_) if index == 1 => ()
      CandidateOperationMismatch if index == 2 => ()
      CandidateCapabilityMismatch if index == 3 => ()
      _ => fail("unexpected candidate identity rejection")
    } noraise {
      _ => fail("hostile candidate identity was admitted")
    }
  }
}

///|
test "candidate ABI role byte alignment count and binding fail closed" {
  let requirements = operand_plan_test_requirements()
  let source = operand_plan_test_source(requirements)
  let layout = operand_plan_test_layout(requirements.source_identity())
  let exact = operand_plan_test_candidate_evidence(requirements)
  let hostile = [
    { ..exact, operand_count: 3, },
    { ..exact, operand_role: HiddenInput, },
    { ..exact, operand_byte_count: 3102716L, },
    { ..exact, operand_alignment: 16L, },
    { ..exact, manifest_bindable: true, },
  ]
  for index, candidate in hostile {
    try plan_from_evidence(source, layout, requirements, candidate, 0) catch {
      OperandOrdinalMismatch if index == 0 => ()
      OperandRoleMismatch if index == 1 => ()
      OperandByteCountMismatch if index == 2 => ()
      OperandAlignmentMismatch if index == 3 => ()
      CandidateUnexpectedlyBindable if index == 4 => ()
      _ => fail("unexpected hostile operand rejection")
    } noraise {
      _ => fail("hostile candidate operand ABI was admitted")
    }
  }
}

///|
test "released source invalid layers and reordered regions fail closed" {
  let requirements = operand_plan_test_requirements()
  let exact_source = operand_plan_test_source(requirements)
  let exact_layout = operand_plan_test_layout(requirements.source_identity())
  let candidate = operand_plan_test_candidate_evidence(requirements)
  try
    plan_from_evidence(
      operand_plan_test_source(requirements, released=true),
      exact_layout,
      requirements,
      candidate,
      0,
    )
  catch {
    SourceReleased => ()
    _ => fail("unexpected released-source rejection")
  } noraise {
    _ => fail("released source was admitted")
  }
  for layer in [-1, 3] {
    try
      plan_from_evidence(
        exact_source, exact_layout, requirements, candidate, layer,
      )
    catch {
      InvalidLayer(actual) => assert_eq(actual, layer)
      _ => fail("unexpected invalid-layer rejection")
    } noraise {
      _ => fail("out-of-range hash layer was admitted")
    }
  }
  try
    plan_from_evidence(
      exact_source,
      operand_plan_test_layout(requirements.source_identity(), source_order=[
        1, 0, 2,
      ]),
      requirements,
      candidate,
      0,
    )
  catch {
    LayoutRegionMismatch(0) => ()
    _ => fail("unexpected reordered-region rejection")
  } noraise {
    _ => fail("reordered sidecar regions were admitted")
  }
}
