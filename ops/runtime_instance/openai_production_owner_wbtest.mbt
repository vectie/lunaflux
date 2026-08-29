///|
test "qualification provenance can never publish production readiness" {
  let valid_ingress = RuntimeOpenAiLoopbackIngressApproval::approve_authenticated_loopback()
  let invalid_ingress : RuntimeOpenAiLoopbackIngressApproval = { marker: 0UL }
  assert_false(
    openai_binding_can_publish_ready(OpenAiQualificationBinding, true, true),
  )
  assert_false(
    openai_binding_can_publish_ready(OpenAiQualificationBinding, false, false),
  )
  assert_false(
    openai_binding_can_publish_ready(
      OpenAiEmbeddingProductionBinding(invalid_ingress),
      true,
      true,
    ),
  )
  assert_false(
    openai_binding_can_publish_ready(
      OpenAiEmbeddingProductionBinding(valid_ingress),
      false,
      false,
    ),
  )
  assert_true(
    openai_binding_can_publish_ready(
      OpenAiEmbeddingProductionBinding(valid_ingress),
      true,
      true,
    ),
  )
  assert_true(
    openai_binding_can_publish_ready(
      OpenAiOpaqueCliProductionBinding,
      false,
      true,
    ),
  )
}

///|
test "OpenAI production binding requires the exact singular lower owner" {
  assert_true(
    openai_binding_state_valid(HealthyNotReady, true, 1, 0, 0, 0, 0, 0),
  )
  assert_false(openai_binding_state_valid(Ready, true, 1, 0, 0, 0, 0, 0))
  assert_false(
    openai_binding_state_valid(OpenAiQualificationReady, true, 1, 0, 0, 0, 0, 0),
  )
  assert_false(
    openai_binding_state_valid(HealthyNotReady, false, 1, 0, 0, 0, 0, 0),
  )
  assert_false(
    openai_binding_state_valid(HealthyNotReady, true, 1, 0, 0, 0, 1, 0),
  )
  assert_false(
    openai_binding_state_valid(HealthyNotReady, true, 1, 0, 0, 0, 0, 1),
  )
}

///|
test "Ready without one healthy server is unhealthy and not ready" {
  let owner = empty_owner_for_state_test(Ready)
  assert_eq(owner.health(), RuntimeUnhealthy)
  assert_eq(owner.readiness(), RuntimeNotReady)
}
