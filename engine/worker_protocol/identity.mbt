///|
/// Monotonic nonzero identity of one submitted schedule plan.
pub struct PlanSequence {
  value : UInt64
} derive(Debug, Eq, Hash)

///|
pub fn PlanSequence::first() -> PlanSequence {
  { value: 1UL }
}

///|
pub fn PlanSequence::from_wire(
  value : UInt64,
) -> PlanSequence raise WorkerProtocolError {
  if value == 0UL {
    raise InvalidIdentity(PlanSequenceIdentity)
  }
  { value, }
}

///|
pub fn PlanSequence::value(self : PlanSequence) -> UInt64 {
  self.value
}

///|
pub fn PlanSequence::next(
  self : PlanSequence,
) -> PlanSequence raise WorkerProtocolError {
  if self.value == 0xffffffffffffffffUL {
    raise IdentityOverflow(PlanSequenceIdentity)
  }
  { value: self.value + 1UL }
}

///|
/// True only for the exact next sequence, never merely a numerically greater
/// sequence.
pub fn PlanSequence::immediately_follows(
  self : PlanSequence,
  previous : PlanSequence,
) -> Bool {
  previous.value != 0xffffffffffffffffUL && self.value == previous.value + 1UL
}

///|
/// Monotonic nonzero identity of the immutable model plan loaded by a worker.
pub struct ModelPlanGeneration {
  value : UInt64
} derive(Debug, Eq, Hash)

///|
pub fn ModelPlanGeneration::first() -> ModelPlanGeneration {
  { value: 1UL }
}

///|
pub fn ModelPlanGeneration::from_wire(
  value : UInt64,
) -> ModelPlanGeneration raise WorkerProtocolError {
  if value == 0UL {
    raise InvalidIdentity(ModelPlanGenerationIdentity)
  }
  { value, }
}

///|
pub fn ModelPlanGeneration::value(self : ModelPlanGeneration) -> UInt64 {
  self.value
}

///|
pub fn ModelPlanGeneration::next(
  self : ModelPlanGeneration,
) -> ModelPlanGeneration raise WorkerProtocolError {
  if self.value == 0xffffffffffffffffUL {
    raise IdentityOverflow(ModelPlanGenerationIdentity)
  }
  { value: self.value + 1UL }
}

///|
pub fn ModelPlanGeneration::immediately_follows(
  self : ModelPlanGeneration,
  previous : ModelPlanGeneration,
) -> Bool {
  previous.value != 0xffffffffffffffffUL && self.value == previous.value + 1UL
}
