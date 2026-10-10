///|
fn admission_test_admit_launches(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  target : @catalog.DeviceTarget,
  launches : ArrayView[JointDiffusionLaunchEvidenceInput],
) -> JointDiffusionArtifactAdmission raise JointDiffusionArtifactAdmissionError {
  admit(
    requirements,
    target~,
    module_inputs=admission_test_modules(),
    entry_point_inputs=admission_test_entries(requirements),
    launch_inputs=launches,
    limits=admission_test_limits(),
  )
}

///|
test "source digest target capability and operation mutations fail closed" {
  let requirements = admission_test_requirements()
  let target = admission_test_target()
  let exact = admission_test_launches(requirements, target)
  let first = requirements.requirements()[0]
  let second = requirements.requirements()[1]
  let wrong_source = exact.copy()
  wrong_source[0] = admission_test_launch(
    target,
    0,
    first.operation(),
    first.capability(),
    admission_test_requirements(fill='b').digest(),
    admission_test_roles(first.operation()),
  )
  try admission_test_admit_launches(requirements, target, wrong_source) catch {
    RequirementDigestMismatch(0) => ()
    _ => fail("unexpected requirement-digest rejection")
  } noraise {
    _ => fail("foreign requirement digest was admitted")
  }
  let wrong_target = exact.copy()
  wrong_target[0] = admission_test_launch(
    admission_test_target(minor=1),
    0,
    first.operation(),
    first.capability(),
    requirements.digest(),
    admission_test_roles(first.operation()),
  )
  try admission_test_admit_launches(requirements, target, wrong_target) catch {
    TargetMismatch(0) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("wrong device target was admitted")
  }
  let wrong_capability = exact.copy()
  wrong_capability[0] = admission_test_launch(
    target,
    0,
    first.operation(),
    second.capability(),
    requirements.digest(),
    admission_test_roles(first.operation()),
  )
  try
    admission_test_admit_launches(requirements, target, wrong_capability)
  catch {
    WrongCapability(0) => ()
    _ => fail("unexpected capability rejection")
  } noraise {
    _ => fail("wrong AOT capability was admitted")
  }
  let wrong_operation = exact.copy()
  wrong_operation[0] = admission_test_launch(
    target,
    0,
    second.operation(),
    first.capability(),
    requirements.digest(),
    admission_test_roles(second.operation()),
  )
  try
    admission_test_admit_launches(requirements, target, wrong_operation)
  catch {
    WrongOperation(0) => ()
    _ => fail("unexpected operation rejection")
  } noraise {
    _ => fail("wrong diffusion operation was admitted")
  }
}

///|
test "wrong or reordered operand ABI fails closed" {
  let requirements = admission_test_requirements()
  let target = admission_test_target()
  let wrong = admission_test_launches(requirements, target)
  let first = requirements.requirements()[0]
  wrong[0] = admission_test_launch(
    target,
    0,
    first.operation(),
    first.capability(),
    requirements.digest(),
    [TextConditioningOutput, TokenIdsInput],
  )
  try admission_test_admit_launches(requirements, target, wrong) catch {
    InvalidLaunchOperands(0) => ()
    _ => fail("unexpected operand rejection")
  } noraise {
    _ => fail("reordered launch operands were admitted")
  }
}

///|
test "missing duplicate and unreferenced entry points fail closed" {
  let requirements = admission_test_requirements()
  let target = admission_test_target()
  let launches = admission_test_launches(requirements, target)
  let exact = admission_test_entries(requirements)
  let missing = exact.copy()
  ignore(missing.pop())
  try
    admit(
      requirements,
      target~,
      module_inputs=admission_test_modules(),
      entry_point_inputs=missing,
      launch_inputs=launches,
      limits=admission_test_limits(),
    )
  catch {
    MissingEntryPoint(entry) => assert_eq(entry, admission_test_entry(33))
    _ => fail("unexpected missing-entry rejection")
  } noraise {
    _ => fail("missing entry point was admitted")
  }
  let duplicate = exact.copy()
  duplicate[1] = duplicate[0]
  try
    admit(
      requirements,
      target~,
      module_inputs=admission_test_modules(),
      entry_point_inputs=duplicate,
      launch_inputs=launches,
      limits=admission_test_limits(),
    )
  catch {
    DuplicateEntryPoint(entry) => assert_eq(entry, admission_test_entry(0))
    _ => fail("unexpected duplicate-entry rejection")
  } noraise {
    _ => fail("duplicate entry point was admitted")
  }
  let unreferenced = exact.copy()
  let foreign = admission_test_entry(50)
  unreferenced.push(
    @artifact.KernelEntryPointInput::new(
      entry_point=foreign,
      function_symbol="joint_diffusion_foreign",
    ),
  )
  try
    admit(
      requirements,
      target~,
      module_inputs=admission_test_modules(),
      entry_point_inputs=unreferenced,
      launch_inputs=launches,
      limits=admission_test_limits(),
    )
  catch {
    UnreferencedEntryPoint(entry) => assert_eq(entry, foreign)
    _ => fail("unexpected unreferenced-entry rejection")
  } noraise {
    _ => fail("unreferenced entry point was admitted")
  }
}

///|
test "distinct requirements cannot reuse one entry-point identity" {
  let requirements = admission_test_requirements()
  let target = admission_test_target()
  let launches = admission_test_launches(requirements, target)
  let second = requirements.requirements()[1]
  launches[1] = try! JointDiffusionLaunchEvidenceInput::new(
    requirements_digest=requirements.digest(),
    requirement_ordinal=1,
    operation=second.operation(),
    capability=second.capability(),
    target~,
    entry_point=admission_test_entry(0),
    dimensions=admission_test_dimensions(),
    operands=admission_test_operands(admission_test_roles(second.operation())),
  )
  try admission_test_admit_launches(requirements, target, launches) catch {
    DuplicateEntryPoint(entry) => assert_eq(entry, admission_test_entry(0))
    _ => fail("unexpected shared-entry rejection")
  } noraise {
    _ => fail("distinct diffusion operations shared one entry point")
  }
}

///|
test "module sets remain bounded but supplied labels do not authenticate bytes" {
  let requirements = admission_test_requirements()
  let target = admission_test_target()
  let launches = admission_test_launches(requirements, target)
  let entries = admission_test_entries(requirements)
  let required_digest = admission_test_module_digest()
  try
    admit(
      requirements,
      target~,
      module_inputs=[],
      entry_point_inputs=entries,
      launch_inputs=launches,
      limits=admission_test_limits(),
    )
  catch {
    MissingModule(digest) => assert_eq(digest, required_digest)
    _ => fail("unexpected missing-module rejection")
  } noraise {
    _ => fail("missing module was admitted")
  }
  let duplicate = admission_test_modules()
  duplicate.push(duplicate[0])
  try
    admit(
      requirements,
      target~,
      module_inputs=duplicate,
      entry_point_inputs=entries,
      launch_inputs=launches,
      limits=admission_test_limits(),
    )
  catch {
    DuplicateModule(digest) => assert_eq(digest, required_digest)
    _ => fail("unexpected duplicate-module rejection")
  } noraise {
    _ => fail("duplicate module was admitted")
  }
  let wrong_bytes = [
    JointDiffusionModuleInput::new(digest=required_digest, module_bytes=b"abd"),
  ]
  let labelled = admit(
    requirements,
    target~,
    module_inputs=wrong_bytes,
    entry_point_inputs=entries,
    launch_inputs=launches,
    limits=admission_test_limits(),
  )
  assert_eq(labelled.modules()[0].digest(), required_digest)
  assert_eq(labelled.modules()[0].module_bytes(), b"abd")
  let foreign_digest = admission_test_digest(b"def")
  let unreferenced = admission_test_modules()
  unreferenced.push(
    JointDiffusionModuleInput::new(digest=foreign_digest, module_bytes=b"def"),
  )
  try
    admit(
      requirements,
      target~,
      module_inputs=unreferenced,
      entry_point_inputs=entries,
      launch_inputs=launches,
      limits=admission_test_limits(),
    )
  catch {
    UnreferencedModule(digest) => assert_eq(digest, foreign_digest)
    _ => fail("unexpected unreferenced-module rejection")
  } noraise {
    _ => fail("unreferenced module was admitted")
  }
}

///|
test "invalid symbol and constructor limits remain bounded" {
  let requirements = admission_test_requirements()
  let target = admission_test_target()
  let entries = admission_test_entries(requirements)
  entries[0] = @artifact.KernelEntryPointInput::new(
    entry_point=admission_test_entry(0),
    function_symbol="bad-symbol",
  )
  try
    admit(
      requirements,
      target~,
      module_inputs=admission_test_modules(),
      entry_point_inputs=entries,
      launch_inputs=admission_test_launches(requirements, target),
      limits=admission_test_limits(),
    )
  catch {
    ArtifactSymbolFailure(_) => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("invalid CUDA symbol was admitted")
  }
  try
    JointDiffusionArtifactAdmissionLimits::new(
      maximum_requirements=0,
      maximum_modules=1,
      maximum_entry_points=1,
      maximum_module_bytes=1L,
      maximum_total_module_bytes=1L,
      maximum_symbol_bytes=1,
      maximum_operands_per_launch=1,
    )
  catch {
    InvalidLimit => ()
    _ => fail("unexpected limit rejection")
  } noraise {
    _ => fail("zero requirement ceiling was admitted")
  }
}
