///|
fn paged_qwen_postprocess_bucket_launch(
  source : @launch_contract.AotLaunchDimensions,
) -> PagedBucketStepLaunch {
  QueryTokenCappedGridX(source)
}

///|
/// The production row-local ABI maps exactly one workgroup to one live token.
/// Reducing the envelope removes inactive workgroups without changing the
/// reduction order, arguments, or memory effects of any live row. Other ABIs
/// retain their admitted geometry rather than inheriting this property.
fn residual_bucket_rule(
  abi : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi,
  source : @launch_contract.AotLaunchDimensions,
  maximum_tokens : Int,
) -> PagedBucketStepLaunch raise DeviceStepError {
  if abi != ResidualRmsNormProductionAbiV2 {
    return FixedBucketLaunch
  }
  if maximum_tokens <= 0 ||
    source.grid_x() != maximum_tokens ||
    source.grid_y() != 1 ||
    source.grid_z() != 1 ||
    source.block_x() != 128 ||
    source.block_y() != 1 ||
    source.block_z() != 1 {
    raise InvalidExecutor(GraphBucket)
  }
  QueryTokenCappedGridX(source)
}

///|
fn prepare_residual_bucket_launches(
  blueprint : PagedKvExecutionBlueprint,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  spans : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan],
  launches : FixedArray[PagedBucketStepLaunch],
) -> Unit raise DeviceStepError {
  for span in spans {
    let (residual, _) = span.operation_indices()
    guard blueprint.steps().get(residual) is Some(operation) else {
      raise InvalidExecutor(GraphBucket)
    }
    let index = qwen_bucket_step_index(steps, operation.operation_id())
    if !(launches[index] is FixedBucketLaunch) {
      raise InvalidExecutor(GraphBucket)
    }
    launches[index] = residual_bucket_rule(
      span.kernel_abi(),
      span.approved_launch(),
      blueprint.profile().max_query_tokens(),
    )
  }
}
