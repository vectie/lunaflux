///|
fn block_control_roles() -> ReadOnlyArray[AdvancedDecoderLaunchOperandRole] {
  [
    StepCountsInput,
    HyperConnectionStateInput,
    HyperConnectionFunctionWeightsInput,
    HyperConnectionBaseInput,
    HyperConnectionScaleInput,
    HyperConnectionPreOutput,
    HyperConnectionPostOutput,
    HyperConnectionCombinationOutput,
  ]
}

///|
fn pre_reduce_roles() -> ReadOnlyArray[AdvancedDecoderLaunchOperandRole] {
  [
    StepCountsInput,
    HyperConnectionStateInput,
    HyperConnectionPreInput,
    HyperConnectionReducedOutput,
  ]
}

///|
fn post_combine_roles() -> ReadOnlyArray[AdvancedDecoderLaunchOperandRole] {
  [
    StepCountsInput,
    HyperConnectionBranchInput,
    HyperConnectionStateInput,
    HyperConnectionPostInput,
    HyperConnectionCombinationInput,
    HyperConnectionCombinedOutput,
  ]
}

///|
fn head_reduce_roles() -> ReadOnlyArray[AdvancedDecoderLaunchOperandRole] {
  [
    StepCountsInput,
    HyperConnectionStateInput,
    HyperConnectionFunctionWeightsInput,
    HyperConnectionBaseInput,
    HyperConnectionScaleInput,
    HyperConnectionReducedOutput,
  ]
}
