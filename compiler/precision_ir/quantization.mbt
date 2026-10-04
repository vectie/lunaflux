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
  { operand: self, passes: ReadOnlyArray::from_array(passes), }
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
  "precision-quantization-v1|\{self.operand.canonical()}|\{passes}"
}
