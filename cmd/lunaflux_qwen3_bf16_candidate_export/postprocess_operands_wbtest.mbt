///|
test "Qwen postprocess consumes projection output without changing its extent" {
  let output = @launch_contract.AotOperand::new(
    role=ActivationOutput(@plan.ActivationRef::new(17)),
    byte_count=786432L,
    alignment=256L,
  )
  let input = qwen_postprocess_activation_input(output)
  assert_eq(input.role(), ActivationInput(@plan.ActivationRef::new(17)))
  assert_eq(input.byte_count(), 786432L)
  assert_eq(input.alignment(), 256L)
}

///|
test "Qwen postprocess rejects a non-output retag" {
  let input = @launch_contract.AotOperand::new(
    role=ActivationInput(@plan.ActivationRef::new(17)),
    byte_count=786432L,
    alignment=256L,
  )
  try qwen_postprocess_activation_input(input) catch {
    QwenIngressPreparation => ()
    _ => fail("unexpected Qwen postprocess operand error")
  } noraise {
    _ => fail("non-output operand reached Qwen postprocess retag")
  }
}
