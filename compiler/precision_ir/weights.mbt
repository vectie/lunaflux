///|
/// Scratch sharing is a lifetime decision, never an implicit dtype property.
pub(all) enum ScratchLifetime {
  Sequential
  Concurrent
} derive(Debug, Eq)

///|
pub extend ScratchLifetime with Eq::{equal, not_equal}

///|
pub extend ScratchLifetime with @debug.Debug::{to_repr}

///|
pub struct WeightBinding {
  priv ordinal : Int
  priv operand : OperandPlan
  priv offset : Int64
} derive(Debug, Eq)

///|
pub extend WeightBinding with Eq::{equal, not_equal}

///|
pub extend WeightBinding with @debug.Debug::{to_repr}

///|
/// Model-wide memory refinement. The ordered ordinal is the model graph's
/// tensor binding, not a family-specific weight name or CUDA address.
pub struct WeightSchema {
  priv bindings : ReadOnlyArray[WeightBinding]
  priv alignment : Int64
  priv resident : Int64
  priv scratch : Int64
  priv lifetime : ScratchLifetime
} derive(Debug, Eq)

///|
pub extend WeightSchema with Eq::{equal, not_equal}

///|
pub extend WeightSchema with @debug.Debug::{to_repr}

///|
pub fn WeightSchema::new(
  operands : ArrayView[OperandPlan],
  alignment_bytes~ : Int64,
  scratch_lifetime~ : ScratchLifetime,
  device_memory_ceiling~ : Int64,
) -> WeightSchema raise PrecisionPlanError {
  if operands.is_empty() ||
    alignment_bytes <= 0L ||
    (alignment_bytes & (alignment_bytes - 1L)) != 0L ||
    device_memory_ceiling < 0L {
    raise IllegalPlacement
  }
  let bindings = []
  let mut resident = 0L
  let mut scratch = 0L
  for ordinal, operand in operands {
    if operand.role != Weight ||
      !(operand.placement
      is (NativeOperand | DecodeInOperandTile | StartupExpandBf16)) {
      raise IllegalPlacement
    }
    let padding = (alignment_bytes - resident % alignment_bytes) %
      alignment_bytes
    if padding > device_memory_ceiling - resident {
      raise MemoryBudgetExceeded
    }
    resident = resident + padding
    if operand.persistent_bytes > device_memory_ceiling - resident {
      raise MemoryBudgetExceeded
    }
    bindings.push({ ordinal, operand, offset: resident, })
    resident = resident + operand.persistent_bytes
    match scratch_lifetime {
      Sequential => scratch = scratch.max(operand.scratch_bytes)
      Concurrent => {
        if operand.scratch_bytes > device_memory_ceiling - scratch {
          raise MemoryBudgetExceeded
        }
        scratch = scratch + operand.scratch_bytes
      }
    }
  }
  if scratch > device_memory_ceiling - resident {
    raise MemoryBudgetExceeded
  }
  {
    bindings: ReadOnlyArray::from_array(bindings),
    alignment: alignment_bytes,
    resident,
    scratch,
    lifetime: scratch_lifetime,
  }
}

///|
pub fn WeightBinding::ordinal(self : WeightBinding) -> Int {
  self.ordinal
}

///|
pub fn WeightBinding::operand(self : WeightBinding) -> OperandPlan {
  self.operand
}

///|
pub fn WeightBinding::offset(self : WeightBinding) -> Int64 {
  self.offset
}

///|
pub fn WeightSchema::bindings(
  self : WeightSchema,
) -> ReadOnlyArray[WeightBinding] {
  self.bindings
}

///|
pub fn WeightSchema::memory(self : WeightSchema) -> (Int64, Int64) {
  (self.resident, self.scratch)
}

///|
pub fn WeightSchema::canonical(self : WeightSchema) -> String {
  let bindings = self.bindings
    .map(b => "\{b.ordinal}:\{b.offset}:\{b.operand.canonical()}")
    .join(";")
  let lifetime = match self.lifetime {
    Sequential => "sequential"
    Concurrent => "concurrent"
  }
  "lunaflux.precision-weights.v1|\{self.alignment}|\{lifetime}|\{self.resident}|\{self.scratch}|\{bindings}"
}
