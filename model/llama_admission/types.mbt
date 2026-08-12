///|
/// Fully validated, allocation-ready dense-Llama startup data.
///
/// This value owns no weight bytes and no device resource. It proves that the
/// selected configuration, safetensors metadata, semantic plan, and exact
/// tensor vocabulary agree before materialization begins.
pub struct LlamaAdmission {
  metadata : @spec.LlamaModelMetadata
  safetensors : @safetensors.SafetensorsMetadata
  plan : @plan.ModelPlan
  bindings : @llama_weights.LlamaWeightBindings
} derive(Debug, Eq)

///|
pub fn LlamaAdmission::metadata(
  self : LlamaAdmission,
) -> @spec.LlamaModelMetadata {
  self.metadata
}

///|
pub fn LlamaAdmission::safetensors(
  self : LlamaAdmission,
) -> @safetensors.SafetensorsMetadata {
  self.safetensors
}

///|
pub fn LlamaAdmission::plan(self : LlamaAdmission) -> @plan.ModelPlan {
  self.plan
}

///|
pub fn LlamaAdmission::bindings(
  self : LlamaAdmission,
) -> @llama_weights.LlamaWeightBindings {
  self.bindings
}
