///|
fn luna_paged_profile_key_after(
  current : LunaPagedKernelProfileObservation,
  previous : LunaPagedKernelProfileObservation,
) -> Bool {
  let current_id = current.operation_id().as_int()
  let previous_id = previous.operation_id().as_int()
  if current_id != previous_id {
    return current_id > previous_id
  }
  String::compare(
    current.shape().digest().as_hex(),
    previous.shape().digest().as_hex(),
  ) >
  0
}

///|
fn luna_paged_profile_from_launches(
  plan : @plan.ModelPlan,
  launches : @launch_contract.PagedKvLaunchContractSet,
  profile_id : @launch_contract.KernelProfileId,
) -> LunaPagedProfileContract raise LunaKernelProfileError {
  let geometry = plan.kv_cache_geometry()
  let layout = launches.device_kv_layout()
  if launches.model_identity() != plan.identity() ||
    layout.identity() != plan.identity() ||
    layout.geometry() != geometry ||
    launches.catalog_version() != @catalog.CatalogVersion::v3() ||
    !launches.device_target().supports_bf16() {
    raise LunaKernelProfileMismatch(Contract)
  }
  let mut selected : @launch_contract.PagedKernelProfile? = None
  for profile in launches.profiles() {
    if profile.id() == profile_id {
      selected = Some(profile)
    }
  }
  guard selected is Some(kernel_profile) else {
    raise LunaKernelProfileMismatch(Contract)
  }
  let mut covered = false
  let selected_launches : Array[@launch_contract.PagedKvAotLaunchContract] = []
  for launch in launches.contracts() {
    if launch.profile().id() == profile_id {
      covered = true
      selected_launches.push(launch)
    }
  }
  if !covered {
    raise LunaKernelProfileMismatch(Contract)
  }
  {
    scope: launches.scope(),
    catalog_version: launches.catalog_version(),
    kernel_profile,
    device_kv_layout: layout,
    launches: ReadOnlyArray::from_array(selected_launches),
    launch_identity: digest_luna_paged_launches(selected_launches),
  }
}

///|
fn luna_paged_launch_covers_operation(
  launches : @launch_contract.PagedKvLaunchContractSet,
  profile_id : @launch_contract.KernelProfileId,
  operation_id : @plan.OperationId,
) -> Bool {
  for launch in launches.contracts() {
    if launch.profile().id() == profile_id &&
      launch.operation_id() == operation_id {
      return true
    }
  }
  false
}

///|
fn luna_paged_validate_shape(
  plan : @plan.ModelPlan,
  contract : LunaPagedProfileContract,
  shape : LunaPagedProfileShape,
) -> Unit raise LunaKernelProfileError {
  let profile = contract.kernel_profile()
  if shape.tokens_per_page() != contract.tokens_per_page() {
    raise LunaKernelProfileMismatch(PageTable)
  }
  if shape.prefill_row_count() + shape.decode_row_count() >
    profile.max_query_rows() ||
    shape.query_token_count() > profile.max_query_tokens() ||
    shape.page_table_block_count() > profile.max_page_table_entries() {
    raise LunaKernelProfileMismatch(Shape)
  }
  let maximum_context = plan.shape_constraints().max_sequence_tokens()
  for row in shape.rows() {
    if row.query_length() > maximum_context ||
      row.context_length() > maximum_context ||
      row.cache_position() >= maximum_context {
      raise LunaKernelProfileMismatch(Shape)
    }
  }
}

///|
/// Admits one offline paged-attention profiler capture. It validates exact
/// live ragged-row and KV geometry, accounts measured time, and chooses one
/// deterministic operation priority. It does not profile, compile, promote,
/// open a device, or establish serving readiness.
pub fn admit_luna_paged_kernel_profile(
  plan : @plan.ModelPlan,
  launches : @launch_contract.PagedKvLaunchContractSet,
  profile_id~ : @launch_contract.KernelProfileId,
  workload~ : LunaKernelProfileDigest,
  profiler~ : LunaKernelProfileDigest,
  trace~ : LunaKernelProfileDigest,
  sampled_wall_time_ns~ : Int64,
  declared_attributed_self_time_ns~ : Int64,
  observations : ArrayView[LunaPagedKernelProfileObservation],
  limits : LunaKernelProfileLimits,
) -> LunaPagedKernelProfileCapture raise LunaKernelProfileError {
  if plan.kv_execution() != PagedKeyValue {
    raise LunaKernelProfileUnsupported(ExecutionMode)
  }
  let contract = luna_paged_profile_from_launches(plan, launches, profile_id)
  let target = launches.device_target()
  if sampled_wall_time_ns <= 0L ||
    sampled_wall_time_ns > LUNA_KERNEL_PROFILE_MAX_NANOS ||
    declared_attributed_self_time_ns <= 0L ||
    declared_attributed_self_time_ns > LUNA_KERNEL_PROFILE_MAX_NANOS {
    raise LunaKernelProfileInvalid(Timing)
  }
  if observations.is_empty() ||
    observations.length() > limits.maximum_observations {
    raise LunaKernelProfileInvalid(Observation)
  }
  let totals = FixedArray::make(plan.operations().length(), 0L)
  let mut actual_total = 0L
  for index, observation in observations {
    guard luna_profile_operation_index(plan, observation.operation_id())
      is Some(operation_index) else {
      raise LunaKernelProfileMismatch(Operation)
    }
    if !luna_paged_launch_covers_operation(
        launches,
        profile_id,
        observation.operation_id(),
      ) {
      raise LunaKernelProfileMismatch(Contract)
    }
    luna_paged_validate_shape(plan, contract, observation.shape())
    if index > 0 &&
      !luna_paged_profile_key_after(observation, observations[index - 1]) {
      raise LunaKernelProfileInvalid(ObservationOrder)
    }
    totals[operation_index] = luna_profile_checked_add(
      totals[operation_index],
      observation.self_time_ns(),
    )
    actual_total = luna_profile_checked_add(
      actual_total,
      observation.self_time_ns(),
    )
  }
  if actual_total != declared_attributed_self_time_ns {
    raise LunaKernelProfileMismatch(Total)
  }
  let priority = luna_profile_priority(plan, totals, actual_total)
  let canonical = encode_luna_paged_kernel_profile(
    plan.identity(),
    target,
    workload,
    profiler,
    trace,
    contract,
    sampled_wall_time_ns,
    actual_total,
    observations,
    limits.maximum_canonical_bytes,
  )
  let digest = LunaKernelProfileDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  {
    identity: plan.identity(),
    target,
    workload,
    profiler,
    trace,
    contract,
    sampled_wall_time_ns,
    attributed_self_time_ns: actual_total,
    observations: ReadOnlyArray::from_array(observations.to_owned()),
    priority,
    canonical,
    digest,
  }
}
