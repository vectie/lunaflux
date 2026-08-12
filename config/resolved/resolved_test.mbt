///|
fn digest() -> String {
  "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}

///|
fn build_plan() -> @resolved.ResolvedPlan {
  let service = try! @service.new("127.0.0.1", 8080, 1048576, 30000, Normal)
  let model = try! @model.new("/models/reference", digest(), [
    "LlamaForCausalLM", "DenseDecoder",
  ])
  let device = try! @device.new([3, 1], 8589934592L)
  @resolved.resolve(service, model, device)
}

///|
test "resolution follows declared stable precedence" {
  let plan = build_plan()
  assert_eq(plan.diagnostic_mode(), Normal)
  assert_eq(plan.selected_architecture(), "LlamaForCausalLM")
  assert_eq(plan.primary_device(), 3)
  assert_eq(plan.memory_ceiling_bytes(), 8589934592L)
  let decisions = plan.decisions()
  assert_eq(decisions.length(), 4)
  assert_eq(decisions[0].subject(), DiagnosticMode)
  assert_eq(decisions[1].reason(), FirstAllowedArchitecture)
  assert_eq(decisions[2].reason(), FirstVisibleDevice)
}

///|
test "identical validated inputs resolve identically" {
  let first = build_plan()
  let second = build_plan()
  assert_eq(first, second)
}
