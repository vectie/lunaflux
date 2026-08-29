///|
fn luna_profile_operation_index(
  plan : @plan.ModelPlan,
  operation_id : @plan.OperationId,
) -> Int? {
  for index, operation in plan.operations() {
    if operation.id() == operation_id {
      return Some(index)
    }
  }
  None
}

///|
fn luna_profile_key_after(
  current : LunaKernelProfileObservation,
  previous : LunaKernelProfileObservation,
) -> Bool {
  let current_id = current.operation_id().as_int()
  let previous_id = previous.operation_id().as_int()
  if current_id != previous_id {
    return current_id > previous_id
  }
  let current_phase = current.shape().phase().as_stable_int()
  let previous_phase = previous.shape().phase().as_stable_int()
  if current_phase != previous_phase {
    return current_phase > previous_phase
  }
  if current.shape().batch_rows() != previous.shape().batch_rows() {
    return current.shape().batch_rows() > previous.shape().batch_rows()
  }
  if current.shape().query_tokens_per_row() !=
    previous.shape().query_tokens_per_row() {
    return current.shape().query_tokens_per_row() >
      previous.shape().query_tokens_per_row()
  }
  current.shape().context_tokens() > previous.shape().context_tokens()
}

///|
fn luna_profile_validate_shape(
  plan : @plan.ModelPlan,
  shape : @device_profile.ExactExecutionShape,
) -> Unit raise LunaKernelProfileError {
  let constraints = plan.shape_constraints()
  if shape.batch_rows() <= 0 ||
    shape.batch_rows() > constraints.max_batch_rows() ||
    shape.query_tokens_per_row() <= 0 ||
    shape.query_tokens_per_row() > constraints.max_sequence_tokens() ||
    shape.context_tokens() <= 0 ||
    shape.context_tokens() > constraints.max_sequence_tokens() {
    raise LunaKernelProfileMismatch(Shape)
  }
}

///|
fn luna_profile_checked_add(
  left : Int64,
  right : Int64,
) -> Int64 raise LunaKernelProfileError {
  if left < 0L || right < 0L || left > LUNA_KERNEL_PROFILE_MAX_NANOS - right {
    raise LunaKernelProfileInvalid(Timing)
  }
  left + right
}

///|
fn luna_profile_priority(
  plan : @plan.ModelPlan,
  totals : FixedArray[Int64],
  attributed_total : Int64,
) -> LunaProfiledOperationPriority raise LunaKernelProfileError {
  let mut selected_index = -1
  let mut selected_time = -1L
  for index, _ in plan.operations() {
    let time = totals[index]
    if time > selected_time {
      selected_index = index
      selected_time = time
    }
  }
  if selected_index < 0 || selected_time <= 0L || attributed_total <= 0L {
    raise LunaKernelProfileInvalid(Total)
  }
  let operation = plan.operations()[selected_index]
  let share = (selected_time * 1_000_000L / attributed_total).to_int()
  {
    operation_id: operation.id(),
    operation_kind: operation.kind(),
    capability: operation.required_capability(),
    attributed_self_time_ns: selected_time,
    attributed_share_ppm: share,
  }
}

///|
pub fn admit_luna_kernel_profile(
  plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  workload~ : LunaKernelProfileDigest,
  profiler~ : LunaKernelProfileDigest,
  sampled_wall_time_ns~ : Int64,
  declared_attributed_self_time_ns~ : Int64,
  observations : ArrayView[LunaKernelProfileObservation],
  limits : LunaKernelProfileLimits,
) -> LunaKernelProfileCapture raise LunaKernelProfileError {
  if plan.kv_execution() != StatelessFullContext {
    raise LunaKernelProfileUnsupported(ExecutionMode)
  }
  if !target.supports_bf16() {
    raise LunaKernelProfileUnsupported(Target)
  }
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
    luna_profile_validate_shape(plan, observation.shape())
    if index > 0 &&
      !luna_profile_key_after(observation, observations[index - 1]) {
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
  let canonical = encode_luna_kernel_profile(
    plan.identity(),
    target,
    workload,
    profiler,
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
    sampled_wall_time_ns,
    attributed_self_time_ns: actual_total,
    observations: ReadOnlyArray::from_array(observations),
    priority,
    canonical,
    digest,
  }
}
