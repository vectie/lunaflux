///|
fn artifact_binding_test_key_value_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  target : @catalog.DeviceTarget,
) -> @advanced_decoder_cuda_aot.AdvancedDecoderKeyValueProjectionCudaCandidate {
  try! @advanced_decoder_cuda_aot.lower_key_value_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=2,
    maximum_tokens=2,
    target~,
    compiler=operand_plan_test_compiler(),
    limits=operand_plan_test_limits(),
  )
}

///|
fn assert_artifact_binding_test_query_b_operation_unavailable(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
) -> Unit raise {
  let expected = 4
  guard requirements.requirements().get(expected) is Some(requirement) else {
    fail("official profile lost its Query-B projection requirement")
  }
  match requirement.operation() {
    QueryLowRankBProjection(_, _, _) => ()
    _ => fail("official profile changed its Query-B operation")
  }
  try artifact_binding_test_admission(requirements, candidate) catch {
    UnsupportedOperation(actual) => assert_eq(actual, expected)
    _ => fail("unexpected unavailable Query-B artifact rejection")
  } noraise {
    _ => fail("token-hash evidence fabricated Query-B projection authority")
  }
}
