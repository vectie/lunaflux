///|
test "routing evidence cannot mask unavailable Query-B projection" {
  let requirements = artifact_test_requirements(hash_layer_count=1)
  let target = artifact_test_target()
  assert_next_artifact_operation_unavailable(
    requirements,
    target,
    artifact_test_launches(requirements, target),
  )
}
