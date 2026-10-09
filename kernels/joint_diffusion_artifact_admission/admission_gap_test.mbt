///|
test "missing and duplicate launches fail closed" {
  let requirements = admission_test_requirements()
  let target = admission_test_target()
  let exact = admission_test_launches(requirements, target)
  let missing = exact.copy()
  ignore(missing.pop())
  try
    admit(
      requirements,
      target~,
      module_inputs=admission_test_modules(),
      entry_point_inputs=admission_test_entries(requirements),
      launch_inputs=missing,
      limits=admission_test_limits(),
    )
  catch {
    MissingLaunch(33) => ()
    _ => fail("unexpected missing-launch rejection")
  } noraise {
    _ => fail("missing launch was admitted")
  }
  let duplicate = exact.copy()
  duplicate[1] = duplicate[0]
  try
    admit(
      requirements,
      target~,
      module_inputs=admission_test_modules(),
      entry_point_inputs=admission_test_entries(requirements),
      launch_inputs=duplicate,
      limits=admission_test_limits(),
    )
  catch {
    DuplicateLaunch(0) => ()
    _ => fail("unexpected duplicate-launch rejection")
  } noraise {
    _ => fail("duplicate launch was admitted")
  }
}
