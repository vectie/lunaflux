///|
test "scheduler publishes its exact ordered A/B plan-buffer identities" {
  let owner = scheduler()
  let binding = owner.plan_buffer_binding()
  assert_eq(binding.count(), 2)
  assert_false(binding.plan_a_identity().same_owner(binding.plan_b_identity()))

  ignore(owner.admit(request(901UL), 0UL))
  let plan_a = next_plan(owner)
  assert_eq(try! plan_a.sequence_wire(), 1UL)
  assert_true(
    binding.plan_a_identity().same_owner(try! plan_a.buffer_identity()),
  )
  assert_false(
    binding.plan_b_identity().same_owner(try! plan_a.buffer_identity()),
  )

  ignore(owner.admit(request(902UL), 0UL))
  let plan_b = next_plan(owner)
  assert_eq(try! plan_b.sequence_wire(), 2UL)
  assert_true(
    binding.plan_b_identity().same_owner(try! plan_b.buffer_identity()),
  )
  assert_false(
    binding.plan_a_identity().same_owner(try! plan_b.buffer_identity()),
  )

  let repeated = owner.plan_buffer_binding()
  assert_eq(repeated.count(), 2)
  assert_true(binding.plan_a_identity().same_owner(repeated.plan_a_identity()))
  assert_true(binding.plan_b_identity().same_owner(repeated.plan_b_identity()))
}
