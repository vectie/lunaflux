///|
fn owner_phase_active_count(phase : I8OwnerSlotPhase) -> Int {
  match phase {
    OwnerSlotsEmpty | ExecutorConstructionOwner => 0
    RawNumericOwner
    | AuthenticatedNumericOwner
    | LoadedNumericOwner
    | ExecutorTransferOwner
    | ReadyExecutorOwner
    | FailedNumericOwner
    | FailedExecutorOwner => 1
  }
}

///|
fn owner_transition_allowed(
  before : I8OwnerSlotPhase,
  after : I8OwnerSlotPhase,
) -> Bool {
  match (before, after) {
    (OwnerSlotsEmpty, RawNumericOwner | FailedNumericOwner)
    | (RawNumericOwner, AuthenticatedNumericOwner)
    | (AuthenticatedNumericOwner, LoadedNumericOwner)
    | (LoadedNumericOwner, ExecutorTransferOwner)
    | (ExecutorTransferOwner, ExecutorConstructionOwner)
    | (
      ExecutorConstructionOwner,
      OwnerSlotsEmpty
      | ReadyExecutorOwner
      | FailedExecutorOwner,
    )
    | (
      RawNumericOwner
      | AuthenticatedNumericOwner
      | LoadedNumericOwner
      | ExecutorTransferOwner
      | ReadyExecutorOwner
      | FailedNumericOwner
      | FailedExecutorOwner,
      OwnerSlotsEmpty,
    ) => true
    _ => false
  }
}

///|
fn active_owner_slot_count(resources : I8DeviceWorkerResources) -> Int {
  (if resources.executor is Some(_) { 1 } else { 0 }) +
  (if resources.numeric is Some(_) { 1 } else { 0 })
}

///|
fn validate_owner_transition(
  expected : I8OwnerSlotPhase,
  observed_before : I8OwnerSlotPhase,
  active_before : Int,
  next : I8OwnerSlotPhase,
  active_after : Int,
) -> Unit raise I8DeviceWorkerError {
  if observed_before != expected ||
    active_before != owner_phase_active_count(expected) ||
    !owner_transition_allowed(expected, next) ||
    active_after != owner_phase_active_count(next) {
    raise Invalid(ResourceState)
  }
}

///|
fn commit_owner_transition(
  resources : I8DeviceWorkerResources,
  expected : I8OwnerSlotPhase,
  next : I8OwnerSlotPhase,
  mutate : () -> Unit,
) -> Unit raise I8DeviceWorkerError {
  let active_before = active_owner_slot_count(resources)
  if resources.owner_phase != expected ||
    active_before != owner_phase_active_count(expected) ||
    !owner_transition_allowed(expected, next) {
    raise Invalid(ResourceState)
  }
  mutate()
  validate_owner_transition(
    expected,
    resources.owner_phase,
    active_before,
    next,
    active_owner_slot_count(resources),
  )
  resources.owner_phase = next
}
