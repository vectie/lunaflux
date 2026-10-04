///|
pub(all) enum OperandRole {
  Weight
  Activation
  KeyCache
  ValueCache
  Output
} derive(Debug, Eq)

///|
pub(all) enum Placement {
  NativeOperand
  DecodeInOperandTile
  StartupExpandBf16
  QuantizeBeforeConsumer
  QuantizeOnCacheWrite
} derive(Debug, Eq)

///|
pub(all) enum NumericEffect {
  Pure
  StartupWrite
  StepWrite
  CacheCommit
} derive(Debug, Eq)

///|
pub(all) suberror PrecisionPlanError {
  IllegalPlacement
  MissingAccuracyContract
  MemoryBudgetExceeded
  ArithmeticOverflow
  NoImplementation
} derive(Debug, Eq)

///|
/// Accuracy is a semantic input to schedule selection, not emitter policy.
pub struct Accuracy {
  priv absolute : Double
  priv relative : Double
} derive(Debug, Eq)

///|
pub fn Accuracy::new(
  absolute~ : Double,
  relative~ : Double,
) -> Accuracy raise PrecisionPlanError {
  if absolute.is_nan() ||
    absolute.is_inf() ||
    relative.is_nan() ||
    relative.is_inf() ||
    absolute < 0.0 ||
    relative < 0.0 {
    raise MissingAccuracyContract
  }
  { absolute, relative, }
}

///|
pub fn Accuracy::bounds(self : Accuracy) -> (Double, Double) {
  (self.absolute, self.relative)
}

///|
pub struct OperandPlan {
  priv role : OperandRole
  priv representation : @format.Representation
  priv placement : Placement
  priv effect : NumericEffect
  priv persistent_bytes : Int64
  priv scratch_bytes : Int64
  priv accuracy : Accuracy
} derive(Debug, Eq)

///|
/// Pure startup/AOT placement pass. Backend-supported placements are supplied
/// explicitly; declaring a dtype never invents native hardware capability.
pub fn plan_operand(
  role~ : OperandRole,
  representation~ : @format.Representation,
  placement~ : Placement,
  supported_placements~ : ArrayView[Placement],
  accuracy~ : Accuracy,
  memory_ceiling~ : Int64,
  tile_elements? : Int = 0,
) -> OperandPlan raise PrecisionPlanError {
  if !supported_placements.contains(placement) {
    raise NoImplementation
  }
  let (rows, columns) = representation.shape()
  let elements = rows.to_int64() * columns.to_int64()
  if elements > 9223372036854775807L / 2L {
    raise ArithmeticOverflow
  }
  let source = representation.total_bytes() catch {
    _ => raise ArithmeticOverflow
  }
  let (effect, persistent_bytes, scratch_bytes) = match (role, placement) {
    (Weight, StartupExpandBf16) => (StartupWrite, elements * 2L, 0L)
    (Weight | Activation | KeyCache | ValueCache, DecodeInOperandTile) => {
      if tile_elements <= 0 {
        raise IllegalPlacement
      }
      (Pure, source, tile_elements.to_int64() * 2L)
    }
    (Activation | Output, QuantizeBeforeConsumer) => (StepWrite, source, 0L)
    (KeyCache | ValueCache, QuantizeOnCacheWrite) => (CacheCommit, source, 0L)
    (_, NativeOperand) => (Pure, source, 0L)
    _ => raise IllegalPlacement
  }
  if memory_ceiling < 0L ||
    persistent_bytes > memory_ceiling ||
    scratch_bytes > memory_ceiling - persistent_bytes {
    raise MemoryBudgetExceeded
  }
  {
    role,
    representation,
    placement,
    effect,
    persistent_bytes,
    scratch_bytes,
    accuracy,
  }
}

///|
pub fn OperandPlan::role(self : OperandPlan) -> OperandRole {
  self.role
}

///|
pub fn OperandPlan::representation(
  self : OperandPlan,
) -> @format.Representation {
  self.representation
}

///|
pub fn OperandPlan::placement(self : OperandPlan) -> Placement {
  self.placement
}

///|
pub fn OperandPlan::effect(self : OperandPlan) -> NumericEffect {
  self.effect
}

///|
pub fn OperandPlan::memory(self : OperandPlan) -> (Int64, Int64) {
  (self.persistent_bytes, self.scratch_bytes)
}

///|
pub fn OperandPlan::accuracy(self : OperandPlan) -> Accuracy {
  self.accuracy
}

///|
pub fn OperandPlan::canonical(self : OperandPlan) -> String {
  let role = match self.role {
    Weight => "weight"
    Activation => "activation"
    KeyCache => "key-cache"
    ValueCache => "value-cache"
    Output => "output"
  }
  let placement = match self.placement {
    NativeOperand => "native"
    DecodeInOperandTile => "tile-decode"
    StartupExpandBf16 => "startup-bf16"
    QuantizeBeforeConsumer => "quantize-consumer"
    QuantizeOnCacheWrite => "quantize-cache"
  }
  "lunaflux.precision-operand.v1|\{role}|\{placement}|\{self.representation.canonical()}|\{self.persistent_bytes}|\{self.scratch_bytes}|\{self.accuracy.absolute}|\{self.accuracy.relative}"
}
