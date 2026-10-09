///|
/// Semantic pass ordering; backend decides launch and reduction geometry.
pub(all) enum QuantizationPass {
  GlobalAmax
  ScaleAmax
  EncodePayload
} derive(Debug, Eq)

///|
pub extend QuantizationPass with Eq::{equal, not_equal}

///|
pub extend QuantizationPass with @debug.Debug::{to_repr}

///|
pub struct QuantizationPlan {
  priv operand : OperandPlan
  priv passes : ReadOnlyArray[QuantizationPass]
  priv amax_floor : Double
} derive(Debug, Eq)

///|
pub extend QuantizationPlan with Eq::{equal, not_equal}

///|
pub extend QuantizationPlan with @debug.Debug::{to_repr}

///|
/// Pure precision pass. Publication of global amax precedes block scaling;
/// a workgroup barrier cannot publish this state across workgroups.
pub fn OperandPlan::quantization(
  self : OperandPlan,
) -> QuantizationPlan raise PrecisionPlanError {
  if self.placement != QuantizeBeforeConsumer &&
    self.placement != QuantizeOnCacheWrite {
    raise IllegalPlacement
  }
  let passes = []
  if self.representation.has_global_scale() {
    passes.push(GlobalAmax)
  }
  if self.representation.scale_element() != Absent {
    passes.push(ScaleAmax)
  }
  passes.push(EncodePayload)
  { operand: self, passes: ReadOnlyArray::from_array(passes), amax_floor: 0.0, }
}

///|
/// A model numeric contract may floor amax before choosing the block scale.
/// This is immutable policy, not a backend's hidden epsilon.
pub fn QuantizationPlan::with_amax_floor(
  self : Self,
  floor : Double,
) -> Self raise PrecisionPlanError {
  if floor.is_nan() ||
    floor.is_inf() ||
    floor < 0.0 ||
    floor > 3.4028234663852886e38 {
    raise MissingAccuracyContract
  }
  { ..self, amax_floor: floor, }
}

///|
pub fn QuantizationPlan::amax_floor(self : Self) -> Double {
  self.amax_floor
}

///|
pub fn QuantizationPlan::operand(self : QuantizationPlan) -> OperandPlan {
  self.operand
}

///|
pub fn QuantizationPlan::passes(
  self : QuantizationPlan,
) -> ReadOnlyArray[QuantizationPass] {
  self.passes
}

///|
pub fn QuantizationPlan::effect(self : QuantizationPlan) -> NumericEffect {
  self.operand.effect
}

///|
pub fn QuantizationPlan::canonical(self : QuantizationPlan) -> String {
  let passes = self.passes
    .map(p => {
      match p {
        GlobalAmax => "global-amax"
        ScaleAmax => "scale-amax"
        EncodePayload => "encode"
      }
    })
    .join(",")
  let identity = "precision-quantization-v1|\{self.operand.canonical()}|\{passes}"
  if self.amax_floor == 0.0 {
    identity
  } else {
    identity + "|amax-floor:\{self.amax_floor}"
  }
}
