///|
/// Joins one digest-pinned reusable CUBIN with an exact attention operation.
/// The CUBIN must export the stable partial/merge symbols selected here. The
/// device owner resolves both symbols before readiness, so an older readonly-
/// only module fails during startup rather than on a token step.
pub fn bind_reusable_paged_attention_decode_split_cuda_aot(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  operands : ArrayView[@launch_contract.AotOperand],
  module_bytes : Bytes,
  expected_module_sha256 : String,
  partial_entry_point_id : @catalog.AotKernelEntryPointId,
  merge_entry_point_id : @catalog.AotKernelEntryPointId,
  maximum_key_value_splits : Int,
  maximum_workspace_bytes : Int64,
) -> PagedAttentionDecodeSplitRuntimeCudaBinding raise PagedAttentionAotError {
  if partial_entry_point_id == merge_entry_point_id ||
    module_bytes.is_empty() ||
    module_bytes.length() > 8388608 ||
    paged_attention_sha256(module_bytes) != expected_module_sha256 ||
    maximum_key_value_splits <= 1 ||
    maximum_workspace_bytes <= 0L {
    raise Invalid(Contract)
  }
  let (query_heads, kv_heads, head_dimension, _, _) = paged_attention_validate_runtime_candidate(
    operation, profile, layout, operands, target,
  )
  if profile.max_page_table_entries() > 2147483647 / layout.tokens_per_page() {
    raise SizeOverflow(Profile)
  }
  let maximum_context_tokens = profile.max_page_table_entries() *
    layout.tokens_per_page()
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads~,
    kv_heads~,
    head_dimension~,
    tokens_per_page=layout.tokens_per_page(),
  ) catch {
    _ => raise Invalid(Shape)
  }
  let dispatch = @luna_attention_strategy.plan_attention_dispatch(
    shape,
    maximum_context_tokens,
    profile.max_query_tokens(),
    maximum_key_value_splits,
  ) catch {
    _ => raise Invalid(Profile)
  }
  let workspace_plan = @luna_attention_strategy.plan_decode_attention_workspace(
    shape,
    dispatch.decode(),
    profile.max_query_rows(),
    maximum_context_tokens,
    maximum_workspace_bytes,
  ) catch {
    _ => raise Unsupported(Profile)
  }
  if workspace_plan.split_count() <= 1 {
    raise Unsupported(Profile)
  }
  let limits = @luna_attention_multilaunch_contract.AttentionMultiLaunchLimits::new(
    maximum_workspace_bytes~,
    maximum_symbol_bytes=256,
    maximum_operands_per_launch=16,
  ) catch {
    _ => raise Invalid(Contract)
  }
  let (maximum_offset, denominator_offset, numerator_offset) = workspace_plan.region_offsets_bytes()
  let workspace = @luna_attention_multilaunch_contract.AttentionDecodeWorkspaceContract::new(
    scalar=F32,
    split_count=workspace_plan.split_count(),
    partial_count=workspace_plan.partial_count(),
    maximum_offset_bytes=maximum_offset,
    denominator_offset_bytes=denominator_offset,
    numerator_offset_bytes=numerator_offset,
    total_bytes=workspace_plan.total_bytes(),
    alignment=16L,
    limits~,
  ) catch {
    _ => raise Invalid(Contract)
  }
  let source_operands = ReadOnlyArray::from_array(operands)
  let (partial_operands, merge_operands) = decode_split_launch_operands(
    source_operands, workspace,
  )
  let module_digest = @catalog.AotArtifactDigest::from_sha256(
    expected_module_sha256,
  ) catch {
    _ => raise Invalid(Contract)
  }
  let family = @catalog.AotKernelFamily::new(
    module_digest~,
    family_id=@catalog.AotKernelFamilyId::from_stable_int(3903) catch {
      _ => raise Invalid(Contract)
    },
  )
  let partial_symbol = "lunaflux_paged_attention_bf16_decode_split_partial_v1_ep_" +
    partial_entry_point_id.as_int().to_string()
  let merge_symbol = "lunaflux_paged_attention_bf16_decode_split_merge_v1_ep_" +
    merge_entry_point_id.as_int().to_string()
  let partial_dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=profile.max_query_rows(),
    grid_y=query_heads,
    grid_z=workspace_plan.split_count(),
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Invalid(Dimensions)
  }
  let merge_dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=profile.max_query_rows(),
    grid_y=query_heads,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Invalid(Dimensions)
  }
  let contract = @luna_attention_multilaunch_contract.admit_attention_decode_multilaunch_contract(
    operation_id=operation.id(),
    profile~,
    target~,
    workspace~,
    partial=@luna_attention_multilaunch_contract.AttentionOrderedLaunchInput::new(
      stage=Partial,
      entry_point=@catalog.AotKernelEntryPoint::new(
        family~,
        entry_point_id=partial_entry_point_id,
      ),
      function_symbol=partial_symbol,
      dimensions=partial_dimensions,
      operands=partial_operands,
    ),
    merge=@luna_attention_multilaunch_contract.AttentionOrderedLaunchInput::new(
      stage=Merge,
      entry_point=@catalog.AotKernelEntryPoint::new(
        family~,
        entry_point_id=merge_entry_point_id,
      ),
      function_symbol=merge_symbol,
      dimensions=merge_dimensions,
      operands=merge_operands,
    ),
    limits~,
  ) catch {
    _ => raise Invalid(Contract)
  }
  {
    operation_id: operation.id(),
    profile,
    layout,
    target,
    contract,
    module_bytes,
  }
}
