///|
/// Compares a positive token ceiling against rows * tokens-per-row without
/// evaluating the potentially overflowing product.
fn fp8_reusable_token_capacity_contains(
  rows : Int,
  tokens_per_row : Int,
  requested : Int,
) -> Bool {
  if rows <= 0 || tokens_per_row <= 0 || requested <= 0 {
    return false
  }
  let quotient = requested / rows
  let remainder = requested % rows
  quotient < tokens_per_row || (quotient == tokens_per_row && remainder == 0)
}

///|
pub fn admit_reusable_paged_v3(
  source : Fp8RuntimeRecipeV2,
  profile : @launch_contract.PagedKernelProfile,
  kv_layout : @device_layout.DeviceKvLayout,
  limits : Fp8ReusablePagedRecipeV3Limits,
) -> Fp8ReusablePagedRecipeV3 raise Fp8ReusablePagedRecipeV3Error {
  let plan = source.startup().model_plan()
  let shape = plan.shape_constraints()
  if kv_layout.identity() != plan.identity() ||
    !kv_layout.is_canonical_for(plan.identity(), plan.kv_cache_geometry()) ||
    kv_layout.stable_layout_version() <= 0 ||
    kv_layout.tokens_per_page() <= 0 {
    raise InvalidFp8ReusablePagedRecipeV3(KvLayout)
  }
  let max_rows = profile.max_query_rows()
  let max_tokens = profile.max_query_tokens()
  let model_rows = shape.max_batch_rows()
  let model_tokens = shape.max_sequence_tokens()
  if max_rows <= 0 ||
    max_tokens <= 0 ||
    max_rows > model_rows ||
    !fp8_reusable_token_capacity_contains(model_rows, model_tokens, max_tokens) ||
    profile.max_page_table_entries() <= 0 ||
    profile.max_page_table_entries() > kv_layout.total_page_count() {
    raise InvalidFp8ReusablePagedRecipeV3(ModelEnvelope)
  }
  let source_launches = source.launches()
  if source_launches.is_empty() ||
    source_launches.length() > limits.max_launches {
    raise InvalidFp8ReusablePagedRecipeV3(LaunchOrder)
  }
  let launches : Array[Fp8ReusablePagedLaunchV3] = []
  let mut workspace_offset = 0L
  for index, launch in source_launches {
    if launch.scope() != PagedFullGraphV1 ||
      launch.profile() != profile.id() ||
      launch.launch_source_version() != PagedNumericV4 ||
      launch.exact_shape().scope() != PagedFullGraphV1 ||
      launch.exact_shape().profile() != Some(profile.id()) ||
      launch.workspace().alignment() != 4L ||
      (
        launch.workspace().byte_length() != 4L &&
        launch.workspace().byte_length() != 8L
      ) {
      raise InvalidFp8ReusablePagedRecipeV3(LaunchAuthority)
    }
    guard plan.operations().get(launch.operation().as_int()) is Some(operation) &&
      operation.id() == launch.operation() &&
      operation.kind() == launch.operation_kind() &&
      plan.numeric_binding().operations().get(launch.operation().as_int())
      is Some(numeric) &&
      numeric.operation() == launch.operation() &&
      numeric.execution().digest() == launch.execution_digest() &&
      numeric.execution().activation_scale() == launch.activation_scale_policy() else {
      raise InvalidFp8ReusablePagedRecipeV3(LaunchAuthority)
    }
    if index > 0 &&
      source_launches[index - 1].operation().as_int() >=
      launch.operation().as_int() {
      raise InvalidFp8ReusablePagedRecipeV3(LaunchOrder)
    }
    let workspace_bytes = launch.workspace().byte_length()
    if workspace_bytes > limits.max_workspace_arena_bytes - workspace_offset {
      raise InvalidFp8ReusablePagedRecipeV3(Workspace)
    }
    launches.push({ source: launch, workspace_offset })
    workspace_offset += workspace_bytes
  }
  let retained = ReadOnlyArray::from_array(launches)
  let canonical = encode_reusable_paged_v3(
    source,
    profile,
    kv_layout,
    retained,
    workspace_offset,
    limits.max_canonical_bytes,
  )
  {
    source,
    profile,
    kv_layout,
    launches: retained,
    workspace_capacity_bytes: workspace_offset,
    canonical,
    digest: Fp8ReusablePagedRecipeV3Digest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
