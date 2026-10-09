///|
test "scaled rotary evidence cannot mask unavailable Query-B projection" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  assert_next_artifact_operation_unavailable(
    requirements,
    target,
    artifact_test_launches(requirements, target),
  )
}
