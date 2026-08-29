///|
test "pointwise lowering rejects swapped manifest operands" {
  let operation = pointwise_residual_operation()
  let operands = [
    pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
    pointwise_operand(ActivationInput(@plan.ActivationRef::new(1)), 64L),
    pointwise_operand(ActivationInput(@plan.ActivationRef::new(0)), 64L),
    pointwise_operand(ActivationOutput(@plan.ActivationRef::new(2)), 64L),
  ]
  try
    @luna_cuda_pointwise_aot.lower_pointwise_cuda_aot_candidate(
      operation,
      pointwise_test_profile(),
      pointwise_test_entry(3),
      pointwise_test_target(),
      pointwise_test_compiler(),
      operands,
      pointwise_test_limits(),
    )
  catch {
    @luna_cuda_pointwise_aot.PointwiseAotError::Mismatch(OperandOrder) => ()
    _ => fail("unexpected rejection")
  } noraise {
    _ => fail("expected positional ABI rejection")
  }
}

///|
test "pointwise lowering rejects non-BF16 target and reassociation" {
  let operation = pointwise_embedding_operation()
  let operands = [
    pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
    pointwise_operand(ExternalInput(TokenIds), 16L, alignment=4L),
    pointwise_operand(WeightInput(@plan.TensorRef::new(0)), 256L),
    pointwise_operand(ActivationOutput(@plan.ActivationRef::new(0)), 64L),
  ]
  for
    case in [
      (pointwise_test_target(supports_bf16=false), pointwise_test_compiler()),
      (pointwise_test_target(), pointwise_test_compiler(reassociate=true)),
    ] {
    let (target, compiler) = case
    try
      @luna_cuda_pointwise_aot.lower_pointwise_cuda_aot_candidate(
        operation,
        pointwise_test_profile(),
        pointwise_test_entry(1),
        target,
        compiler,
        operands,
        pointwise_test_limits(),
      )
    catch {
      @luna_cuda_pointwise_aot.PointwiseAotError::Unsupported(_) => ()
      _ => fail("unexpected rejection")
    } noraise {
      _ => fail("expected policy rejection")
    }
  }
}

///|
test "content addressed output path rejects non-digests" {
  assert_eq(
    @luna_cuda_pointwise_aot.content_addressed_cubin_path(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    "sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.cubin",
  )
  try
    @luna_cuda_pointwise_aot.content_addressed_cubin_path("not-a-digest")
  catch {
    Invalid(Digest) => ()
    _ => fail("unexpected rejection")
  } noraise {
    _ => fail("expected digest rejection")
  }
}
