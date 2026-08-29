///|
test "plan-buffer startup binding is exact and epoch-sequence neutral" {
  let owner = wb_owner_with_capacity(3, 3)
  let binding = owner.plan_buffer_binding()
  assert_true(physical_equal(binding, owner.plan_buffer_binding))
  assert_true(binding.plan_a_identity().same_owner(owner.plan_a.identity()))
  assert_true(binding.plan_b_identity().same_owner(owner.plan_b.identity()))

  ignore(owner.admit(wb_request(911UL), 0UL))
  ignore(owner.admit(wb_request(912UL), 0UL))
  ignore(owner.admit(wb_request(913UL), 0UL))
  let plan_a = wb_next_plan(owner)
  let plan_b = wb_next_plan(owner)
  assert_true(
    binding.plan_a_identity().same_owner(try! plan_a.buffer_identity()),
  )
  assert_true(
    binding.plan_b_identity().same_owner(try! plan_b.buffer_identity()),
  )
  assert_eq(try! plan_a.sequence_wire(), 1UL)
  assert_eq(try! plan_b.sequence_wire(), 2UL)

  ignore(owner.complete(plan_a, wb_final_completion(owner, plan_a, 17)))
  try plan_a.buffer_identity() catch {
    _ => ()
  } noraise {
    _ => fail("retired A submission remained current")
  }
  let plan_three = wb_next_plan(owner)
  assert_eq(try! plan_three.sequence_wire(), 3UL)
  assert_true(
    binding.plan_a_identity().same_owner(try! plan_three.buffer_identity()),
  )
  assert_true(physical_equal(binding, owner.plan_buffer_binding()))
  assert_eq(binding.count(), 2)
}
