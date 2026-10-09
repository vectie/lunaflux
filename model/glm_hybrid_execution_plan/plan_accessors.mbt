///|
pub fn GlmHybridExecutionPlan::identity(self : Self) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn GlmHybridExecutionPlan::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn GlmHybridExecutionPlan::profile_identity(self : Self) -> String {
  self.profile_identity
}

///|
pub fn GlmHybridExecutionPlan::architecture(self : Self) -> String {
  self.architecture
}

///|
pub fn GlmHybridExecutionPlan::model_type(self : Self) -> String {
  self.model_type
}

///|
pub fn GlmHybridExecutionPlan::vocabulary_size(self : Self) -> Int {
  self.vocabulary_size
}

///|
pub fn GlmHybridExecutionPlan::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn GlmHybridExecutionPlan::dense_intermediate_size(self : Self) -> Int {
  self.dense_intermediate_size
}

///|
pub fn GlmHybridExecutionPlan::decoder_layer_count(self : Self) -> Int {
  self.decoder_layer_count
}

///|
pub fn GlmHybridExecutionPlan::context_length(self : Self) -> Int {
  self.context_length
}

///|
pub fn GlmHybridExecutionPlan::sparse_attention(
  self : Self,
) -> GlmDynamicSparseAttentionPlan {
  self.sparse_attention
}

///|
pub fn GlmHybridExecutionPlan::kda(self : Self) -> GlmKdaExecutionPlan? {
  self.kda
}

///|
pub fn GlmHybridExecutionPlan::hyper_connection(
  self : Self,
) -> @advanced_decoder_execution_plan.HyperConnectionPlan? {
  self.hyper_connection
}

///|
pub fn GlmHybridExecutionPlan::moe(
  self : Self,
) -> @advanced_decoder_execution_plan.MoeExecutionPlan {
  self.moe
}

///|
pub fn GlmHybridExecutionPlan::moe_router_float32(self : Self) -> Bool {
  self.moe_router_float32
}

///|
pub fn GlmHybridExecutionPlan::moe_correction_bias(self : Self) -> Bool {
  self.moe_correction_bias
}

///|
pub fn GlmHybridExecutionPlan::layers(
  self : Self,
) -> ReadOnlyArray[GlmHybridLayerPlan] {
  self.layers
}

///|
pub fn GlmHybridExecutionPlan::auxiliary_prediction(
  self : Self,
) -> @advanced_decoder_execution_plan.AuxiliaryPredictionPlan {
  self.auxiliary_prediction
}

///|
pub fn GlmHybridExecutionPlan::vision_conditioning(
  self : Self,
) -> GlmVisionConditioningPlan? {
  self.vision_conditioning
}
