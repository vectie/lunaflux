///|
fn runtime_bundle_test_launch(
  block_x : Int32,
  grid_y : Int32,
  shared_memory_bytes : Int32,
  grid_z~ : Int32 = 1,
) -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y~,
    grid_z~,
    block_x~,
    block_y=1,
    block_z=1,
    shared_memory_bytes~,
  )
}

///|
fn runtime_bundle_test_input(
  abi : @luna_fused_artifact_admission.ReusableFusedRuntimeKernelAbi,
  first_operation : Int32,
  operation_count : Int32,
  suffix : String,
) -> @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput raise @luna_fused_artifact_admission.FusedArtifactAdmissionError {
  @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
    abi~,
    first_operation~,
    operation_count~,
    function_symbol="lunaflux_" + suffix,
    launch=runtime_bundle_test_launch(
      if abi == QwenPagedAttentionReadOnlyProductionV2 ||
        abi == QwenPagedAttentionReadOnlyDecodeSplitProductionV3 {
        32
      } else {
        128
      },
      if abi == QwenQkNormPositionedRopePagedKvWriteProductionV2 {
        8
      } else {
        1
      },
      if abi == ResidualRmsNormProductionV2 {
        512
      } else {
        0
      },
    ),
    module_bytes=@utf8.encode("cubin-" + suffix),
  )
}
