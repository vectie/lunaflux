///|
fn device_sampling_abi_bytes(
  rows : Int,
  row_bytes : Int64,
) -> Int64 raise SamplingBoundaryError {
  if rows <= 0 ||
    row_bytes <= 0L ||
    rows.to_int64() > 9223372036854775807L / row_bytes {
    raise SizeOverflow(DeviceSamplingParameters)
  }
  rows.to_int64() * row_bytes
}

///|
/// Returns the exact future binding surface without granting current manifest
/// authority. No model, artifact, module, or device resource is opened.
pub fn device_sampling_cuda_abi_v2(
  plan : @luna_sampling_strategy.SamplingDevicePlan,
) -> DeviceSamplingCudaAbiV2 raise SamplingBoundaryError {
  if plan.maximum_rows() <= 0 ||
    plan.parameter_row_bytes() != 40L ||
    plan.result_row_bytes() != 8L ||
    plan.workspace_bytes() <= 0L {
    raise Invalid(DeviceSamplingPlan)
  }
  {
    function_symbol: device_sampling_cuda_function_symbol(),
    operands: [
      SamplingCounts,
      SamplingRowOffsets,
      SamplingBf16Logits,
      SamplingParametersV2,
      SamplingWorkspace,
      SamplingResults,
    ],
    grid_x: plan.maximum_rows(),
    block_x: 256,
    parameter_bytes: device_sampling_abi_bytes(
      plan.maximum_rows(),
      plan.parameter_row_bytes(),
    ),
    workspace_bytes: plan.workspace_bytes(),
    result_bytes: device_sampling_abi_bytes(
      plan.maximum_rows(),
      plan.result_row_bytes(),
    ),
  }
}
