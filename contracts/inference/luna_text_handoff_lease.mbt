///|
fn LunaTextRequestHandoffStorage::require_handoff(
  self : Self,
  epoch : UInt64,
  claimed : Bool,
) -> Unit raise InferenceContractError {
  let expected = if claimed {
    LUNA_TEXT_HANDOFF_CLAIMED
  } else {
    LUNA_TEXT_HANDOFF_LEASED
  }
  if epoch == 0UL || self.epoch != epoch || self.phase != expected {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
}

///|
pub fn LunaTextRequestHandoffLease::take_claim(
  self : Self,
) -> LunaTextRequestHandoffClaim raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.phase = LUNA_TEXT_HANDOFF_CLAIMED
  { storage: self.storage, epoch: self.epoch }
}

///|
pub fn LunaTextRequestHandoffLease::inference_limits(
  self : Self,
) -> InferenceLimits raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.limits
}

///|
pub fn LunaTextRequestHandoffLease::model_identity(
  self : Self,
) -> @spec.ModelIdentity raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.model_identity
}

///|
pub fn LunaTextRequestHandoffLease::deadline(
  self : Self,
) -> DeadlineBudget raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.deadline
}

///|
pub fn LunaTextRequestHandoffLease::request_id(
  self : Self,
) -> RequestId raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.request_id
}

///|
pub fn LunaTextRequestHandoffLease::effective_limits(
  self : Self,
) -> EffectiveLimits raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.effective
}

///|
pub fn LunaTextRequestHandoffLease::validated_sampling(
  self : Self,
) -> LunaValidatedSampling raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.sampling
}

///|
pub fn LunaTextRequestHandoffLease::sampling_seed(
  self : Self,
) -> SamplingSeed raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.sampling_seed
}

///|
pub fn LunaTextRequestHandoffLease::stream_preference(
  self : Self,
) -> StreamPreference raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.stream_preference
}

///|
pub fn LunaTextRequestHandoffLease::prompt_length(
  self : Self,
) -> Int raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.prompt_length
}

///|
pub fn LunaTextRequestHandoffLease::prompt_byte_at(
  self : Self,
  index : Int,
) -> Byte raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  if index < 0 || index >= self.storage.prompt_length {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, index)
  }
  self.storage.prompt[index]
}

///|
/// Exposes the already-validated stop and cache projection without transferring
/// its release authority. This view becomes stale with this handoff lease.
pub fn LunaTextRequestHandoffLease::semantic_view(
  self : Self,
) -> LunaRequestSemanticView raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.semantic_leases[0].view()
}

///|
pub fn LunaTextRequestHandoffLease::release(
  self : Self,
) -> Unit raise InferenceContractError {
  self.storage.require_handoff(self.epoch, false)
  self.storage.semantic_leases[0].release()
  self.storage.semantic_leases.clear()
  self.storage.reset_scalars()
  self.storage.phase = LUNA_TEXT_HANDOFF_IDLE
}

///|
fn LunaTextRequestHandoffClaim::require(
  self : Self,
) -> LunaTextRequestHandoffStorage raise InferenceContractError {
  self.storage.require_handoff(self.epoch, true)
  self.storage
}

///|
pub fn LunaTextRequestHandoffClaim::prompt_length(
  self : Self,
) -> Int raise InferenceContractError {
  self.require().prompt_length
}

///|
pub fn LunaTextRequestHandoffClaim::prompt_byte_at(
  self : Self,
  index : Int,
) -> Byte raise InferenceContractError {
  let storage = self.require()
  if index < 0 || index >= storage.prompt_length {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, index)
  }
  storage.prompt[index]
}

///|
pub fn LunaTextRequestHandoffClaim::take_semantics(
  self : Self,
) -> LunaRequestSemanticLease raise InferenceContractError {
  let storage = self.require()
  if storage.semantic_leases.length() != 1 {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, 0)
  }
  let lease = storage.semantic_leases[0]
  storage.semantic_leases.clear()
  lease
}

///|
pub fn LunaTextRequestHandoffClaim::release(
  self : Self,
) -> Unit raise InferenceContractError {
  let storage = self.require()
  if storage.semantic_leases.length() == 1 {
    storage.semantic_leases[0].release()
    storage.semantic_leases.clear()
  }
  storage.reset_scalars()
  storage.phase = LUNA_TEXT_HANDOFF_IDLE
}
