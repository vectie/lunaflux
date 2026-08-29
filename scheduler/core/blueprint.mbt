///|
fn blueprint_worker_limits(
  capacity : @runtime_resolved.ResolvedRuntimeCapacity,
) -> @worker_protocol.WorkerProtocolLimits raise SchedulerError {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=capacity.worker_max_prefill_rows(),
    max_decode_rows=capacity.worker_max_decode_rows(),
    max_plan_rows=capacity.worker_max_plan_rows(),
    max_plan_tokens=capacity.worker_max_plan_tokens(),
    max_plan_pages=capacity.worker_max_plan_pages(),
    max_capabilities=capacity.worker_max_capabilities(),
    max_completion_slots=capacity.worker_max_completion_slots(),
    max_sequence_tokens=capacity.worker_max_sequence_tokens(),
    max_token_id=capacity.worker_max_token_id(),
  ) catch {
    error => raise WorkerProtocol(error)
  }
}

///|
pub fn SchedulerBlueprint::new(
  capacity : @runtime_resolved.ResolvedRuntimeCapacity,
  model_identity : @spec.ModelIdentity,
  model_plan_generation : @worker_protocol.ModelPlanGeneration,
  intermediate_prefill_recipe : @worker_protocol.RowCapabilityRecipe,
  final_prefill_recipe : @worker_protocol.RowCapabilityRecipe,
  decode_recipe : @worker_protocol.RowCapabilityRecipe,
) -> SchedulerBlueprint raise SchedulerError {
  if intermediate_prefill_recipe.identity() != model_identity ||
    final_prefill_recipe.identity() != model_identity ||
    decode_recipe.identity() != model_identity ||
    intermediate_prefill_recipe.model_plan_generation() != model_plan_generation ||
    final_prefill_recipe.model_plan_generation() != model_plan_generation ||
    decode_recipe.model_plan_generation() != model_plan_generation {
    raise invalid(ModelIdentity, Mismatch)
  }
  let prefill_max = if intermediate_prefill_recipe.length() >
    final_prefill_recipe.length() {
    intermediate_prefill_recipe.length()
  } else {
    final_prefill_recipe.length()
  }
  let max_recipe_length = if prefill_max > decode_recipe.length() {
    prefill_max
  } else {
    decode_recipe.length()
  }
  if capacity.max_rows_per_step() >
    capacity.worker_max_capabilities() / max_recipe_length {
    raise invalid(Plan, OutOfRange)
  }
  {
    capacity,
    model_identity,
    model_plan_generation,
    intermediate_prefill_recipe,
    final_prefill_recipe,
    decode_recipe,
    worker_limits: blueprint_worker_limits(capacity),
  }
}

///|
pub fn SchedulerBlueprint::model_identity(
  self : SchedulerBlueprint,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn SchedulerBlueprint::model_plan_generation(
  self : SchedulerBlueprint,
) -> @worker_protocol.ModelPlanGeneration {
  self.model_plan_generation
}

///|
pub fn SchedulerBlueprint::worker_protocol_limits(
  self : SchedulerBlueprint,
) -> @worker_protocol.WorkerProtocolLimits {
  self.worker_limits
}

///|
/// Returns the fixed request-slot count allocated by `instantiate`.
pub fn SchedulerBlueprint::request_slot_capacity(
  self : SchedulerBlueprint,
) -> Int {
  self.capacity.total_request_slots()
}

///|
/// Allocates a fresh, independent scheduler owner.
pub fn SchedulerBlueprint::instantiate(
  self : SchedulerBlueprint,
) -> Scheduler raise SchedulerError {
  Scheduler::new(
    self.capacity,
    self.model_identity,
    self.model_plan_generation,
    self.intermediate_prefill_recipe,
    self.final_prefill_recipe,
    self.decode_recipe,
  )
}
